from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

Path("models").mkdir(exist_ok=True)

dataset_path = Path("data/prepared_mfcc_dataset.npz")

with np.load(dataset_path) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

print("MFCC dataset loaded successfully!")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("\nShapes after adding channel dimension:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


model = keras.Sequential([
    layers.Input(shape=(20, 130, 1)),

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

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="models/best_mfcc_music_genre_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("MFCC Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("MFCC Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

best_model = keras.models.load_model(
    "models/best_mfcc_music_genre_model.keras"
)

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nMFCC Test Results:")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

y_pred_prob = best_model.predict(X_test)

y_pred = np.argmax(y_pred_prob, axis=1)

print("\nMFCC Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock"
        ]
    )
)

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "blues", "classical", "country", "disco",
        "hiphop", "jazz", "metal", "pop", "reggae", "rock"
    ],
    yticklabels=[
        "blues", "classical", "country", "disco",
        "hiphop", "jazz", "metal", "pop", "reggae", "rock"
    ]
)

plt.title("MFCC Model - Confusion Matrix")
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.tight_layout()
plt.show()