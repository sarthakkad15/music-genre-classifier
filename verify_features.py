from pathlib import Path
import numpy as np

features_path = Path("data/processed_features.npz")

with np.load(features_path) as data:
    features = data["features"]
    labels = data["labels"]
    processed_files = data["processed_files"]

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
print("Processed files shape:", processed_files.shape)

assert features.shape == (999, 128, 130)
assert len(labels) == len(processed_files) == len(features)

print("\nSaved data loaded successfully")
print("First genre: ", labels[0])
print("First file: ", processed_files[0])