from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

features_path = Path("data/mfcc_features.npz")

with np.load(features_path) as data:
    features = data["features"]
    labels = data["labels"]
    processed_files = data["processed_files"]

print("MFCC dataset loaded:", features.shape)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
    features,
    encoded_labels,
    test_size=0.30,
    random_state=42,
    stratify=encoded_labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

output_path = Path("data/prepared_mfcc_dataset.npz")

np.savez_compressed(
    output_path,
    X_train=np.asarray(X_train, dtype=np.float32),
    X_val=np.asarray(X_val, dtype=np.float32),
    X_test=np.asarray(X_test, dtype=np.float32),
    y_train=y_train,
    y_val=y_val,
    y_test=y_test
)

print("\nPrepared MFCC dataset saved:", output_path.resolve())

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)