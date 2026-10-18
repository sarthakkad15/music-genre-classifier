from pathlib import Path

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

print("MUSIC GENRE CLASSIFIER - DATASET INSPECTION\n")

if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH.resolve()}"
    )

total_files = 0

for genre in GENRES:
    genre_path = DATASET_PATH / genre
    if not genre_path.is_dir():
        print(f"{genre:10} | FOLDER MISSING")
        continue

    files = sorted(
        path for path in genre_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    count = len(files)
    total_files += count

    print(f"{genre:10} | {count:3} WAV files")

    if files:
        print(f"  Example file: {files[0].name}")


print("\nTotal WAV files in dataset:", total_files)
print("Expected genres:", len(GENRES))