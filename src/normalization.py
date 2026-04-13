import numpy as np


def normalize(img, desired_mean=100.0, desired_var=100.0):
    
    img = img.astype(np.float32)

    mean = np.mean(img)
    var = np.var(img)

    if var < 1e-8:
        return img.copy()

    normalized = np.where(
        img > mean,
        desired_mean + np.sqrt((desired_var * (img - mean) ** 2) / var),
        desired_mean - np.sqrt((desired_var * (img - mean) ** 2) / var),
    )

    return normalized.astype(np.float32)