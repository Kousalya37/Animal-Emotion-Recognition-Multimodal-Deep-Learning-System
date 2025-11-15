import numpy as np

def simulate(num_samples):
    hr = np.random.randint(60, 90, num_samples)
    temp = np.random.uniform(36.0, 38.0, num_samples)
    variability = np.random.uniform(0.7, 1.0, num_samples)
    return np.vstack((hr, temp, variability)).T

if __name__ == "__main__":
    n = 100  
    feats = simulate(n)
    np.save("biometric_features.npy", feats)
    print(f"Saved {n} biometric samples.")
