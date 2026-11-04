import numpy as np


def add_gaussian_noise(image, sigma, rng=None):
    """Add Gaussian noise to an image.

    image: float32 array with values in [0, 1]
    sigma: noise std on the 0-255 scale (e.g. 25)
    rng:   optional np.random.Generator, for reproducibility
    """
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma / 255.0, image.shape).astype(np.float32) #0 (mean), scale, size
    return np.clip(image + noise, 0.0, 1.0)