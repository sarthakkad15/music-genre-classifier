import librosa
import numpy as np

SAMPLE_RATE = 22050

audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)

pitch_audio = librosa.effects.pitch_shift(
    y=audio,
    sr=SAMPLE_RATE,
    n_steps=1
)

stretched_audio = librosa.effects.time_stretch(
    y=audio,
    rate=1.05
)

print("Original audio shape:", audio.shape)
print("Pitch audio shape:", pitch_audio.shape)
print("Time stretched shape:", stretched_audio.shape)

print("\nAugmentation functions working!")