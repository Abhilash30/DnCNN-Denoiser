import numpy as np
from scipy.ndimage import gaussian_filter

import config
from src.dataset import DenoiseDataset
from src.metrics import psnr

ds = DenoiseDataset(f"{config.DATA_DIR}/val", config.PATCH_SIZE, config.SIGMA, train=False)

for blur in [0, 0.5, 1.0, 1.5, 2.0]:
    scores = []
    for i in range(len(ds)):
        noisy, clean = ds[i]
        out = gaussian_filter(noisy[0].numpy(), sigma=blur)
        scores.append(psnr(out, clean[0].numpy()))
    print(f"blur {blur}: {np.mean(scores):.2f} dB")