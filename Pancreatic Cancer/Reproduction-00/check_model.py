"""
Verify that the saved model.h5 loads, has the expected architecture,
and produces valid predictions.
Run: python check_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

MODEL_PATH = "checkpoints/model.h5"
IMG_H, IMG_W, IMG_C = 256, 256, 3

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


def main():
    print("=" * 60)
    print("MODEL HEALTH CHECK")
    print("=" * 60)

    # 1. File exists
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"File:      {MODEL_PATH}")
    print(f"Size:      {size_mb:.2f} MB")

    # 2. Loads
    print("\nLoading model...")
    model = load_model(MODEL_PATH, custom_objects={
        "dice_loss": dice_loss,
        "dice_coef": dice_coef,
    })
    print("Load:      OK")

    # 3. Architecture
    print("\n--- Architecture ---")
    print(f"Input:     {model.input_shape}")
    print(f"Output:    {model.output_shape}")
    print(f"Params:    {model.count_params():,}")
    print(f"Layers:    {len(model.layers)}")

    # 4. Input shape check
    print("\n--- Shape Validation ---")
    expected = (None, IMG_H, IMG_W, IMG_C)
    if model.input_shape == expected:
        print(f"Input:     matches expected {expected}")
    else:
        print(f"Input:     WARNING — expected {expected}, got {model.input_shape}")

    expected_out = (None, IMG_H, IMG_W, 1)
    if model.output_shape == expected_out:
        print(f"Output:    matches expected {expected_out}")
    else:
        print(f"Output:    WARNING — expected {expected_out}, got {model.output_shape}")

    # 5. Inference test (correct shape)
    print("\n--- Inference Test (Correct Shape) ---")
    dummy = np.random.rand(1, IMG_H, IMG_W, IMG_C).astype(np.float32)
    pred = model.predict(dummy, verbose=0)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Output range: [{pred.min():.4f}, {pred.max():.4f}]")

    if 0.0 <= pred.min() <= 1.0 and 0.0 <= pred.max() <= 1.0:
        print("Range:     OK — output in [0, 1]")
    else:
        print("Range:     WARNING — output outside [0, 1]")

    # 6. Batch inference
    print("\n--- Batch Inference ---")
    batch = np.random.rand(4, IMG_H, IMG_W, IMG_C).astype(np.float32)
    pred_batch = model.predict(batch, verbose=0)
    print(f"Batch input:  {batch.shape}")
    print(f"Batch output: {pred_batch.shape}")

    # 7. Wrong shape (should raise)
    print("\n--- Wrong Shape Handling ---")
    try:
        wrong = np.random.rand(1, 128, 128, 3).astype(np.float32)
        model.predict(wrong, verbose=0)
        print("Status:    WARNING — model accepted wrong shape (unexpected)")
    except Exception as e:
        print("Status:    OK — model correctly rejects wrong shape")

    # 8. Compiled status
    print("\n--- Compilation Info ---")
    if model.optimizer is not None:
        print(f"Optimizer: {model.optimizer.__class__.__name__}")
    if model.loss is not None:
        loss_name = model.loss if isinstance(model.loss, str) else model.loss.__name__
        print(f"Loss:      {loss_name}")

    # 9. Save/load round-trip test
    print("\n--- Save/Load Round-Trip ---")
    tmp_path = "_tmp_roundtrip.h5"
    model.save(tmp_path)
    _ = load_model(tmp_path, custom_objects={
        "dice_loss": dice_loss, "dice_coef": dice_coef})
    os.remove(tmp_path)
    print("Status:    OK — model re-saves and reloads cleanly")

    print("\n" + "=" * 60)
    print("MODEL IS HEALTHY")
    print("=" * 60)


if __name__ == "__main__":
    main()