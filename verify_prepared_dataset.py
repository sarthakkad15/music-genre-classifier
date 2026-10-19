from pathlib import Path
import numpy as np

dataset_path = Path("data/prepared_dataset.npz")

with np.load(dataset_path) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    processed_files = data["processed_files"]

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Test features:", X_test.shape)

print("\nTraining labels:", y_train.shape)
print("Validation labels:", y_val.shape)
print("Test labels:", y_test.shape)

assert X_train.shape == (699, 128, 130)
assert X_val.shape == (150, 128, 130)
assert X_test.shape == (150, 128, 130)

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_test) == len(y_test)

all_labels = np.concatenate([y_train, y_val, y_test])

assert len(np.unique(all_labels)) == 10
assert len(processed_files) == 999

print("\nAll verification checks passed")