"""
Train the U-Net model for pancreatic tumor segmentation.
Run: python train.py
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     MaxPool2D, Conv2DTranspose, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, CSVLogger,
                                        ReduceLROnPlateau, EarlyStopping)
from tensorflow.keras import backend as K

# ---------------- Hyperparameters ----------------
H, W = 256, 256
BATCH_SIZE = 16
LR = 1e-4
NUM_EPOCHS = 20
SEED = 42

DATA_DIR = "./data"
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
CKPT_DIR = "checkpoints"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(CKPT_DIR, "model.h5")
CSV_PATH = os.path.join(RESULTS_DIR, "log.csv")

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- Loss ----------------
smooth = 1e-15


def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred + smooth)
    return numerator / denominator


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# ---------------- Model ----------------
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model


# ---------------- Data ----------------
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = (x > 0.5).astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


def load_paths():
    images = sorted(glob(os.path.join(IMG_DIR, "*")))
    masks = sorted(glob(os.path.join(MASK_DIR, "*")))

    if len(images) == 0 or len(masks) == 0:
        raise ValueError(
            f"No files found in {IMG_DIR} or {MASK_DIR}. "
            "Add images to data/images and masks to data/masks."
        )

    split_size = int(len(images) * 0.02)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=SEED)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=SEED)
    return (train_x, train_y), (valid_x, valid_y)


# ---------------- Train ----------------
if __name__ == "__main__":
    create_dir(CKPT_DIR)
    create_dir(RESULTS_DIR)

    (train_x, train_y), (valid_x, valid_y) = load_paths()
    print(f"Train: {len(train_x)} images | Valid: {len(valid_x)} images")

    train_dataset = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH_SIZE)

    model = build_unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(LR), metrics=["accuracy"])
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5,
                          min_lr=1e-7, verbose=1),
        CSVLogger(CSV_PATH),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ]

    history = model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=valid_dataset,
        callbacks=callbacks,
    )

    # Save training curves
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Dice Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150)
    plt.close()

    print(f"\nTraining complete. Best model: {MODEL_PATH}")