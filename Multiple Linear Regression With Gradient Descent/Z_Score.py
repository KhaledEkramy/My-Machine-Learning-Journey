import numpy as np

def zscore_normalize_features(X):
    mue = np.mean(X)
    sigma = np.std(X)
    
    X_normalized = (X - mue) / sigma
    
    return (X_normalized, mue, sigma)