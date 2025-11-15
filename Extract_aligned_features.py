import numpy as np

audio = np.load("audio_sequence_features.npy")
image = np.load("image_features_fast.npy")
bio = np.load("biometric_features.npy")

min_len = min(len(audio), len(image), len(bio))
print(f"Aligning features to {min_len} samples.")

audio = audio[:min_len]
image = image[:min_len]
bio = bio[:min_len]

np.save("audio_features_aligned.npy", audio)
np.save("image_features_aligned.npy", image)
np.save("biometric_features_aligned.npy", bio)
