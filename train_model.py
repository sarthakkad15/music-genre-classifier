from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = Path("data/prepared_dataset.npz")

with np.load(dataset_path) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

print("Dataset loaded successfully!")

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Test features:", X_test.shape)

print("Training labels:", y_train.shape)
print("Validation labels:", y_val.shape)
print("Test labels:", y_test.shape)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("\nShapes after adding channel dimension:")
print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Test features:", X_test.shape)

model = keras.Sequential([
    layers.Input(shape=(128,130, 1)),

    layers.Conv2D(
        filters=32, 
        kernel_size=(3, 3), 
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        units=256,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        units=10,
        activation="softmax"
    )
])
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)

print("\nFinal training results:")

print("Training Accuracy:", history.history["accuracy"][-1])
print("Validation Accuracy:", history.history["val_accuracy"][-1])
print("Training Loss:", history.history["loss"][-1])
print("Validation Loss:", history.history["val_loss"][-1])
