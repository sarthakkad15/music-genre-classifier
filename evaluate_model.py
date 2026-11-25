from pathlib import Path
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

dataset_path = Path("data/prepared_dataset.npz")

with np.load(dataset_path) as data:
    X_val = data["X_val"]
    y_val = data["y_val"]

X_val = np.expand_dims(X_val, axis=-1)

model_path = Path("models/best_music_genre_model.keras")

model = keras.models.load_model(model_path)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)

print("\nBest Checkpoint Results:")
print("Validation Accuracy:", val_accuracy)
print("Validation Loss:", val_loss)

y_pred_prob = model.predict(X_val)

y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_val, y_pred)

genre_names = [
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

report = classification_report(
    y_val,
    y_pred,
    target_names=genre_names
)

print("\nClassification Report:")
print(report)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=genre_names, 
    yticklabels=genre_names
)

plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.title("Music Genre Classification - Confusion Matrix")

plt.tight_layout()
plt.show()