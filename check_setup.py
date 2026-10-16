
import librosa
import numpy
import pandas
import matplotlib
import sklearn
import tensorflow as tf
import soundfile

print("All libraries imported successfully!")
print("Librosa:", librosa.__version__)
print("NumPy:", numpy.__version__)
print("TensorFlow:", tf.__version__)
print("GPU detected:", len(tf.config.list_physical_devices("GPU")) > 0)
