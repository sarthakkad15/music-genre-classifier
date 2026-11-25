from pathlib import Path
import librosa
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def augment_audio(audio,sr):
    augmentation_type = np.random.choice(["pitch", "stretch", "shift"])

    if augmentation_type == "pitch":
        n_steps = np.random.uniform(-1, 1)
        augmented_audio = librosa.effects.pitch_shift(
            y=audio, 
            sr=sr, 
            n_steps=n_steps
        )

    elif augmentation_type == "stretch":
        rate = np.random.uniform(0.95, 1.05)
        augmented_audio = librosa.effects.time_stretch(
            y=audio, 
            rate=rate
        )
    else:
        shift = np.random.randint(
            -int(0.1*sr),
            int(0.1*sr)
        )
        augmented_audio = np.zeros_like(audio)

        if shift > 0:
            augmented_audio[shift:] = audio[:-shift]
        elif shift < 0:
            augmented_audio[:shift] = audio[-shift:]
        else:
            augmented_audio = audio.copy()


    if len(augmented_audio) < SAMPLES_PER_TRACK:
        augmented_audio = np.pad(
            augmented_audio, 
            (0, SAMPLES_PER_TRACK - len(augmented_audio))
        )
    else:
        augmented_audio = augmented_audio[:SAMPLES_PER_TRACK]
    
    return augmented_audio

features_path = Path("data/processed_features.npz")

with np.load(features_path) as data:
    features = data["features"]
    labels = data["labels"]
    processed_files = data["processed_files"]

print("Original dataset loaded: ",features.shape)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

train_indices, temp_indices = train_test_split(
    np.arange(len(features)),
    test_size=0.30,
    random_state=42,
    stratify=encoded_labels
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=encoded_labels[temp_indices]
)

print("\nOriginal split reproduced:")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Testing:", len(test_indices))

X_train = features[train_indices]
y_train = encoded_labels[train_indices]

X_val = features[val_indices]
y_val = encoded_labels[val_indices]

X_test = features[test_indices]
y_test = encoded_labels[test_indices]

print("\nTraining features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Test features:", X_test.shape)

manifest_path = Path("dataset_manifest.csv")
df = pd.read_csv(manifest_path)

augmented_features = []
augmented_labels = []

for index in train_indices:
    audio_path = Path(processed_files[index])
    genre_label = encoded_labels[index]

    audio, sr = librosa.load(
        audio_path, 
        sr=SAMPLE_RATE, 
        mono=True, 
        duration=DURATION
    )

    if len(audio) < SAMPLES_PER_TRACK:
        audio = np.pad(
            audio,
            (0, SAMPLES_PER_TRACK - len(audio))
        )
    else:
        audio = audio[:SAMPLES_PER_TRACK]

    augmented_audio = augment_audio(audio, sr)

    mel = librosa.feature.melspectrogram(
        y=augmented_audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    augmented_features.append(mel_db)
    augmented_labels.append(genre_label)

    if len(augmented_features) % 100 == 0:
        print(f"Augmented {len(augmented_features)} / {len(train_indices)} songs")

X_augmented = np.asarray(augmented_features,dtype=np.float32)

y_augmented = np.asarray(augmented_labels,dtype=np.int64)

X_train_final = np.concatenate(
    [X_train, X_augmented],
    axis=0
)

y_train_final = np.concatenate(
    [y_train, y_augmented],
    axis=0
)

print("\nFinal dataset sizes:")
print("Training:", X_train_final.shape)
print("Training Labels:", y_train_final.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

assert X_train_final.shape == (1398, 128, 130)
assert len(X_train_final) == len(y_train_final)
assert X_val.shape == (150, 128, 130)
assert X_test.shape == (150, 128, 130)

print("\nAugmentation pipeline completed successfully!")

output_path = Path("data/augmented_dataset.npz")

np.savez_compressed(
    output_path,
    X_train=X_train_final,
    y_train=y_train_final,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

print("\nAugmented dataset saved to:", output_path.resolve())