from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = Path("data/genres_original")

audio_files = sorted((DATASET_PATH/"blues").glob("*.wav"))

if not audio_files:
    raise FileNotFoundError(
        f"No WAV files found in {DATASET_PATH / 'blues'}. "
        "Check your dataset path."
    )

audio_path = audio_files[0]

print("Audio file:", audio_path)

audio, sr = librosa.load(audio_path, sr=22050, mono=True)

duration = len(audio) / sr

print("Sampling rate:", sr, "Hz")
print("Number of samples:", len(audio))
print("Duration:", round(duration, 2), "seconds")

plt.figure(figsize=(12, 4))

librosa.display.waveshow(audio, sr=sr)

plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Mel-spectrogram
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel",
    cmap="magma"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Mel-Spectrogram")
plt.tight_layout()
plt.show()
