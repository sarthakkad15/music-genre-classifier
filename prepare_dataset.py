from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

features_path = Path("data/processed_features.npz")

with np.load(features_path) as data:
    features = data["features"]
    labels = data["labels"]
    processed_files = data["processed_files"]

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

print("\nData split completed successfully")
print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))
print("Test examples:", len(X_test))

print("Genre-to-number mapping:")

for number, genre in enumerate(label_encoder.classes_):
    print(f"{genre} -> {number}")

print("\nOriginal labels:", labels[:10])
print("Encoded labels:", encoded_labels[:10])

print("\nFeatures shape:", features.shape)
print("Encoded labels shape:", encoded_labels.shape)
print("Number of classes:", len(label_encoder.classes_))

assert len(features) == len(encoded_labels) == len(processed_files)
assert len(label_encoder.classes_) == 10

print("\nLabel encoding completed successfully")

for name, y in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(f"\n{name} genre distribution:")

    genres, counts = np.unique(
        label_encoder.inverse_transform(y), 
        return_counts=True
    )

    for genre, count in zip(genres, counts):
        print(f"{genre} : {count}")

output_path = Path("data/prepared_dataset.npz")

np.savez_compressed(
    output_path,
    X_train=np.asarray(X_train, dtype=np.float32),
    X_val=np.asarray(X_val, dtype=np.float32),
    X_test=np.asarray(X_test, dtype=np.float32),
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    processed_files=processed_files
)

print("\nPrepared dataset saved successfully to:", output_path.resolve())