import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout
)
from tensorflow.keras.models import Sequential

# -------------------------------
# Dataset paths
# -------------------------------
TRAIN_DIR = "train"
VAL_DIR = "valid"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# -------------------------------
# Load datasets
# -------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# -------------------------------
# Build CNN model
# -------------------------------
model = Sequential([
    Conv2D(32, 3, padding="same", activation="relu", input_shape=(128,128,3)),
    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(64, 3, padding="same", activation="relu"),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(128, 3, padding="same", activation="relu"),
    Conv2D(128, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(256, 3, padding="same", activation="relu"),
    Conv2D(256, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(512, 3, padding="same", activation="relu"),
    Conv2D(512, 3, activation="relu"),
    MaxPooling2D(),

    Dropout(0.25),
    Flatten(),
    Dense(1500, activation="relu"),
    Dropout(0.4),
    Dense(38, activation="softmax")
])

# -------------------------------
# Compile
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Train
# -------------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -------------------------------
# Save model (IMPORTANT)
# -------------------------------
model.save("trained_model", save_format="tf")

print("✅ Model training complete and saved as 'trained_model/'")
