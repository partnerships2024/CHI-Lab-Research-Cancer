"""
Evaluate the trained U-Net model on the test set and produce in-distribution results.
Run: python test.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# ---------------- Parameters ----------------
H, W = 256, 256
SEED = 42
DATA_DIR = "./data"
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
MODEL_PATH = "checkpoints/model.h5"
RESULTS_DIR = "results"
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
THRESHOLD = 0.5

smooth = 1e-15


# ---------------- Custom objects ----------------
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred + smooth)
    return numerator / denominator


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# ---------------- Helpers ----------------
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((H, 10, 3)) * 255
    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


# ---------------- Evaluation ----------------
def main():
    create_dir(RESULTS_DIR)
    create_dir(PRED_DIR)

    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH, custom_objects={
        "dice_loss": dice_loss,
        "dice_coef": dice_coef,
    })

    # Build validation set (same split as training)
    images = sorted(glob(os.path.join(IMG_DIR, "*")))
    masks = sorted(glob(os.path.join(MASK_DIR, "*")))

    split_size = int(len(images) * 0.02)
    _, valid_x = train_test_split(images, test_size=split_size, random_state=SEED)
    _, valid_y = train_test_split(masks, test_size=split_size, random_state=SEED)

    if len(valid_x) == 0:
        raise ValueError("No validation images found. Ensure data/ is populated.")

    print(f"Evaluating on {len(valid_x)} images...")

    SCORE = []
    for x_path, y_path in tqdm(zip(valid_x, valid_y), total=len(valid_x)):
        name = os.path.basename(x_path)

        image = cv2.imread(x_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (W, H))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        mask = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))

        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = (y_pred >= THRESHOLD).astype(np.int32)

        save_image_path = os.path.join(PRED_DIR, name)
        save_results(image, mask, y_pred, save_image_path)

        mask_bin = (mask / 255.0 > 0.5).astype(np.int32).flatten()
        y_pred_flat = y_pred.flatten()

        d_loss = dice_loss(mask_bin, y_pred_flat).numpy() if hasattr(dice_loss(mask_bin, y_pred_flat), "numpy") else dice_loss(mask_bin, y_pred_flat)
        f1_value = f1_score(mask_bin, y_pred_flat, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask_bin, y_pred_flat, labels=[0, 1], average="binary")
        recall_value = recall_score(mask_bin, y_pred_flat, labels=[0, 1],
                                    average="binary", zero_division=0)
        precision_value = precision_score(mask_bin, y_pred_flat, labels=[0, 1],
                                          average="binary", zero_division=0)

        SCORE.append([name, f1_value, jac_value, float(d_loss),
                      recall_value, precision_value])

    if not SCORE:
        raise ValueError("No predictions were produced.")

    # Aggregate
    arr = np.array([s[1:] for s in SCORE], dtype=np.float64)
    mean_scores = arr.mean(axis=0)  # f1, iou, dice_loss, recall, precision

    dice_mean = 1.0 - mean_scores[2]

    print("\n" + "=" * 55)
    print("IN-DISTRIBUTION TEST RESULTS")
    print("=" * 55)
    print(f"  F1        : {mean_scores[0]:.5f}")
    print(f"  IoU       : {mean_scores[1]:.5f}")
    print(f"  Dice      : {dice_mean:.5f}")
    print(f"  Recall    : {mean_scores[3]:.5f}")
    print(f"  Precision : {mean_scores[4]:.5f}")
    print("=" * 55)

    # Save per-image CSV
    df = pd.DataFrame(
        SCORE,
        columns=["Image", "F1", "IoU", "dice_loss", "Recall", "Precision"],
    )
    df.to_csv(os.path.join(RESULTS_DIR, "scores.csv"), index=False)

    # Save summary CSV
    summary = pd.DataFrame([{
        "dice": dice_mean,
        "f1": mean_scores[0],
        "iou": mean_scores[1],
        "recall": mean_scores[3],
        "precision": mean_scores[4],
    }])
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"), index=False)

    # Metrics bar plot
    plt.figure(figsize=(10, 5))
    labels = ["Dice", "F1", "IoU", "Recall", "Precision"]
    values = [dice_mean, mean_scores[0], mean_scores[1],
              mean_scores[3], mean_scores[4]]
    plt.bar(labels, values, color="steelblue")
    plt.ylim(0, 1)
    plt.title("In-Distribution Evaluation Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_barplot.png"), dpi=150)
    plt.close()

    # Sample predictions
    sample_n = min(3, len(valid_x))
    fig, axes = plt.subplots(sample_n, 3, figsize=(12, 4 * sample_n))
    if sample_n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(sample_n):
        img = cv2.imread(valid_x[i], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (W, H))
        gt = cv2.imread(valid_y[i], cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (W, H))
        pred = model.predict(np.expand_dims(img / 255.0, 0), verbose=0)[0]
        pred = (np.squeeze(pred, -1) >= THRESHOLD).astype(np.uint8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")
        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"), dpi=150)
    plt.close()

    print(f"\nAll results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()