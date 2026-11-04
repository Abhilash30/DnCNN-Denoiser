import numpy as np
from src.noise import add_gaussian_noise

rng = np.random.default_rng(0)
clean = np.full((256, 256, 3), 0.5, dtype=np.float32)  # flat gray image
noisy = add_gaussian_noise(clean, sigma=25, rng=rng)

diff = noisy - clean
print(diff.mean(), diff.std() * 255)

import torch
from src.model import DnCNN

m = DnCNN()
print(sum(p.numel() for p in m.parameters()))
print(m(torch.rand(16, 1, 64, 64)).shape)