from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


MANIFEST_PATH = Path("dataset_manifest.csv")

df=pd.read_csv(MANIFEST_PATH)

features = []
labels = []
processed_files = []
failed_files = []

for index, row in df.iterrows():
    audio_path = Path(row["file_path"])
    genre = row["genre"]

    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    except Exception as error:
        print(f"Failed to load: {audio_path}")
        print(f"Error: {error}")
        failed_files.append({"file_path": audio_path, "error": str(error)})
        continue

    if len(audio) < SAMPLES_PER_TRACK:
        audio = np.pad(audio, (0, SAMPLES_PER_TRACK - len(audio)))
    else:
        audio = audio[:SAMPLES_PER_TRACK]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    features.append(mel_db)
    labels.append(genre)
    processed_files.append(str(audio_path))
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} / {len(df)} songs")

print("\nFeature extraction complete")
print("Total features:", len(features))
print("Total labels:", len(labels))
print("Shape of one feature:", features[0].shape)
print("Failed files:", len(failed_files))
print("Processed files:", len(processed_files))

assert len(features) == len(labels) == len(processed_files), (
    "Features, labels, and processed files are not aligned!"
)

assert all(feature.shape == (128, 130) for feature in features), (
    "Some features have an unexpected shape!"
)

print("All validation checks passed!")

output_path = Path("data/processed_features.npz")

np.savez_compressed(
    output_path,
    features=np.array(features, dtype=np.float32),
    labels=np.array(labels),
    processed_files=np.array(processed_files)
)

print("\nExtracted data saved successfully to:", output_path.resolve())