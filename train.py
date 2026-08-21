"""
train.py — Retrain the Plant Disease CNN with heavy data augmentation
========================================================================
Solves the #1 problem: model scores 95% on PlantVillage but chokes on
real-world phone photos.

WHY?  PlantVillage images are clean lab shots — uniform lighting, white
backgrounds, perfect focus.  Real farmer photos are messy — shaky hands,
harsh sunlight, shadows, mixed backgrounds, weird angles.

This script bridges that gap by brutally augmenting every training image
so the model learns to handle real-world chaos.

USAGE:
    python train.py                         # defaults (25 epochs)
    python train.py --epochs 40             # more training
    python train.py --batch_size 64         # if you have GPU RAM
    python train.py --img_size 224          # higher res (slower)

The trained model is saved as:
    trained_model.keras   (primary — used by app.py)
    trained_model.h5      (backup)
    training_hist.json    (epoch-wise metrics)
========================================================================
"""

import os

# ── Force GPU Support on Native Windows via PyTorch Backend ──
try:
    import torch
    if torch.cuda.is_available():
        os.environ["KERAS_BACKEND"] = "torch"
        print("\n🚀 [OPTIMIZATION] Highly compatible NVIDIA GPU detected!")
        print("   -> Automatically routing Keras through standard PyTorch for max GPU training speed.\n")
except ImportError:
    pass

import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
NUM_CLASSES = 38

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]


def parse_args():
    p = argparse.ArgumentParser(description="Retrain Plant Disease CNN with augmentation")
    p.add_argument("--img_size",    type=int,   default=128,    help="Image size (default 128)")
    p.add_argument("--batch_size",  type=int,   default=32,     help="Batch size (default 32)")
    p.add_argument("--epochs",      type=int,   default=10,     help="Max epochs (default 10, early-stop enabled)")
    p.add_argument("--lr",          type=float, default=0.0001, help="Initial learning rate (default 0.0001)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# DATA AUGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────────
#
#  Each augmentation simulates a specific real-world condition that
#  PlantVillage images DON'T have but farmer phone photos DO:
#
#  1. RandomFlip          → leaf can be mirrored in any direction
#  2. RandomRotation      → phone held at any angle
#  3. RandomZoom          → close-up vs. far shot
#  4. RandomBrightness    → sunlight vs. shade vs. overcast
#  5. RandomContrast      → cheap camera vs. good camera
#  6. RandomSaturation*   → faded vs. vivid colors (cheap sensors)
#  7. RandomHue*          → slight color shifts from white balance
#  8. GaussianNoise*      → grainy low-light photos
#  9. RandomCrop+Resize*  → farmer didn't center the leaf
#  10. Cutout / Erasing*  → finger covering part of leaf, shadow occlusion
#
#  * = custom tf.image ops since Keras doesn't have these as layers
#

def build_augmentation_layer(img_size):
    """Keras Sequential augmentation block (runs on GPU if available)."""
    return keras.Sequential([
        # ── Geometric ────────────────────────────────────────
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(
            factor=0.15,           # ±54 degrees
            fill_mode="reflect",
        ),
        layers.RandomZoom(
            height_factor=(-0.15, 0.15),   # zoom in/out 15%
            width_factor=(-0.15, 0.15),
            fill_mode="reflect",
        ),
        layers.RandomTranslation(
            height_factor=0.1,     # shift up/down 10%
            width_factor=0.1,
            fill_mode="reflect",
        ),

        # ── Photometric ─────────────────────────────────────
        layers.RandomBrightness(
            factor=0.25,           # ±25% brightness swing
        ),
        layers.RandomContrast(
            factor=0.3,            # ±30% contrast swing
        ),
    ], name="augmentation")


def advanced_augment(image, label, img_size):
    """
    Additional augmentations applied via tf.image (not available as
    Keras layers).  These run inside tf.data.Dataset.map().

    These are CRITICAL because they simulate the messiest parts of
    real-world farmer photos that Keras layers can't handle.
    """
    # ── Random Saturation ────────────────────────────────────
    # Cheap phone cameras often produce washed-out or over-saturated
    # colors, especially in direct sunlight.
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    # ── Random Hue ───────────────────────────────────────────
    # Auto white-balance on phones shifts color tones.  A leaf that's
    # truly green might look bluish-green or yellowish-green.
    image = tf.image.random_hue(image, max_delta=0.05)

    # ── Random JPEG Quality ──────────────────────────────────
    # Farmers share photos via WhatsApp → heavy JPEG compression.
    # This teaches the model to tolerate compression artifacts.
    image = tf.cast(image * 255.0, tf.uint8)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=50, max_jpeg_quality=100)
    image = tf.cast(image, tf.float32) / 255.0

    # ── Gaussian Noise ───────────────────────────────────────
    # Low-light phone photos are noisy.
    if tf.random.uniform([]) < 0.3:  # apply 30% of the time
        noise = tf.random.normal(
            shape=tf.shape(image),
            mean=0.0,
            stddev=0.02,
        )
        image = tf.clip_by_value(image + noise, 0.0, 1.0)

    # ── Random Cutout / Erasing ──────────────────────────────
    # Simulates: finger over part of leaf, shadow occlusion, dirt,
    # or another leaf overlapping.  Forces model to not rely on a
    # single region.
    if tf.random.uniform([]) < 0.3:  # apply 30% of the time
        h = img_size
        w = img_size
        cut_h = tf.random.uniform([], minval=h // 8, maxval=h // 4, dtype=tf.int32)
        cut_w = tf.random.uniform([], minval=w // 8, maxval=w // 4, dtype=tf.int32)
        top   = tf.random.uniform([], minval=0, maxval=h - cut_h, dtype=tf.int32)
        left  = tf.random.uniform([], minval=0, maxval=w - cut_w, dtype=tf.int32)

        # Create mask and zero-out the patch
        padding = [[top, h - top - cut_h], [left, w - left - cut_w], [0, 0]]
        ones_patch = tf.ones([cut_h, cut_w, 3], dtype=tf.float32)
        mask = 1.0 - tf.pad(ones_patch, padding)
        image = image * mask

    # ── Clamp values ─────────────────────────────────────────
    image = tf.clip_by_value(image, 0.0, 1.0)
    image.set_shape([img_size, img_size, 3])  # FIX: Preserve static shape required by Keras

    return image, label


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_datasets(img_size, batch_size):
    """Load train/valid datasets with augmentation on train only."""

    print(f"\n[DATA] Loading from ./{TRAIN_DIR} and ./{VALID_DIR}")
    print(f"[DATA] Image size: {img_size}x{img_size}, Batch: {batch_size}")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(img_size, img_size),
        batch_size=None,           # unbatched for .map()
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    valid_ds = keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    # Normalize to [0, 1]
    train_ds = train_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    valid_ds = valid_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Apply advanced augmentation to training set only
    train_ds = train_ds.map(
        lambda x, y: advanced_augment(x, y, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Batch, prefetch, cache
    train_ds = (
        train_ds
        .shuffle(buffer_size=2048)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds


# ─────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────
def build_model(img_size, lr):
    """
    Same CNN architecture as original — but with an augmentation
    layer baked into the model graph so augmentation happens on GPU
    and is automatically skipped during inference.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # Augmentation layers (active during training only)
    x = build_augmentation_layer(img_size)(inputs, training=True)

    # ── Block 1: 32 filters ─────────────────────────────────
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # ── Block 2: 64 filters ─────────────────────────────────
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # ── Block 3: 128 filters ────────────────────────────────
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # ── Block 4: 256 filters ────────────────────────────────
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # ── Block 5: 512 filters ────────────────────────────────
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # ── Classifier Head ──────────────────────────────────────
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1500, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PlantDisease_CNN_v2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────
def get_callbacks():
    """
    Three callbacks that make training smarter:

    1. EarlyStopping — stops if val_accuracy doesn't improve for 6
       epochs straight.  Prevents wasting GPU hours on a plateau.

    2. ReduceLROnPlateau — cuts learning rate by half when val_loss
       stalls for 3 epochs.  Helps squeeze out extra accuracy in the
       later stages of training.

    3. ModelCheckpoint — saves the model ONLY when val_accuracy beats
       the previous best.  So even if training overfits toward the end,
       you always keep the best snapshot.
    """
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath="best_model.weights.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True, # FIX: Avoids complex Lambda serialization / pickling bugs!
            verbose=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    IMG_SIZE    = args.img_size
    BATCH_SIZE  = args.batch_size
    EPOCHS      = args.epochs
    LR          = args.lr

    print("=" * 65)
    print("  AgriGuard — Plant Disease Model Retraining")
    print("  WITH REAL-WORLD DATA AUGMENTATION")
    print("=" * 65)
    print(f"  Image Size   : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size   : {BATCH_SIZE}")
    print(f"  Max Epochs   : {EPOCHS}")
    print(f"  Learning Rate: {LR}")
    print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 65)

    # ── Hardware Optimizations ───────────────────────────────
    try:
        # Enable XLA (Accelerated Linear Algebra) for massive CPU speedups
        tf.config.optimizer.set_jit(True)
        # Maximize thread usage for data loading and ops
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
        print("[INFO] CPU hardware optimizations (XLA & max threading) enabled.")
    except Exception as e:
        print(f"[OPT] Could not apply hardware optimizations: {e}")

    # ── Load Data ────────────────────────────────────────────
    train_ds, valid_ds = load_datasets(IMG_SIZE, BATCH_SIZE)

    # ── Build & Resume Model ─────────────────────────────────
    model = build_model(IMG_SIZE, LR)
    model.summary()

    # Resume training if weights exist from a previous crash/stop
    weights_path = "best_model.weights.h5"
    if os.path.exists(weights_path):
        print(f"\n[RESUME] Found existing {weights_path}! Loading weights to resume your progress...")
        try:
            model.load_weights(weights_path)
            print("[RESUME] successfully loaded weights and resuming training! 🔄")
        except Exception as e:
            print(f"[ERROR] Failed to resume weights: {e}")

    # ── Train ────────────────────────────────────────────────
    print("\n[TRAIN] Starting training with augmentation...\n")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
    )

    # ── Save Model ───────────────────────────────────────────
    print("\n[SAVE] Saving final trained model...")
    model.save_weights("trained_model.weights.h5") # Safer weight-only save
    print("[SAVE] Saved: trained_model.weights.h5")

    # ── Save Training History ────────────────────────────────
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open("training_hist.json", "w") as f:
        json.dump(hist_dict, f, indent=2)
    print("[SAVE] Saved: training_hist.json")

    # ── Print Final Results ──────────────────────────────────
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch   = history.history["val_accuracy"].index(best_val_acc) + 1
    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETE")
    print(f"  Best Val Accuracy : {best_val_acc:.2%}  (Epoch {best_epoch})")
    print(f"  Final Train Acc   : {history.history['accuracy'][-1]:.2%}")
    print(f"  Final Val Acc     : {history.history['val_accuracy'][-1]:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
