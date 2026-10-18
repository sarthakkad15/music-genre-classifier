from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/genres_original")

GENRES = [
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

records = []

for genre in GENRES:
    genre_path = DATASET_PATH / genre
    if not genre_path.is_dir():
        raise FileNotFoundError(
            f"Genre folder not found: {genre_path.resolve()}"
        )

    audio_files = sorted(
        path for path in genre_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )

    for audio_file in audio_files:
        records.append({
            "file_path": audio_file.as_posix(),
            "filename": audio_file.name,
            "genre": genre
        })

df=pd.DataFrame(records)

if df.empty:
    raise ValueError("No WAV files found in the dataset.")

output_path = Path("dataset_manifest.csv")
df.to_csv(output_path, index=False)

print("Dataset inventory created successfully.")
print("Total audio files: ", len(df))
print("\nFiles per genre")
print(df["genre"].value_counts().sort_index())
print("\nFirst five records:")
print(df.head())
print(f"\nsaved to: {output_path.resolve()}")
