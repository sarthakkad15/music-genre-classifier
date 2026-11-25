import numpy as np

data = np.load("data/processed_features.npz")

features = data["features"]

print("Features shape:", features.shape)
print("Minimum value: ", features.min())
print("Maximum value: ", features.max())
print("Mean value: ", features.mean())
print("Data type: ", features.dtype)

assert features.min() >=0
assert features.max() <= 1

print("\nNormalization verified successfully!")