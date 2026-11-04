import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.noise import add_gaussian_noise


class DenoiseDataset(Dataset):
    def __init__(self, folder, patch_size, sigma, train=True,
                 patches_per_image=16, seed=0):
        extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(folder, ext)))
        paths = sorted(paths)
        
        print(f"Loaded {len(paths)} images from {folder}") 
        
        self.images = [
            np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            for p in paths
        ]
        
        self.patch_size = patch_size
        self.sigma = sigma
        self.train = train
        self.patches_per_image = patches_per_image
        self.seed = seed

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + idx)

        h, w = img.shape
        p = self.patch_size
        top = rng.integers(0, h - p + 1)
        left = rng.integers(0, w - p + 1)
        clean = img[top:top + p, left:left + p]

        if self.train:
            if rng.random() < 0.5:
                clean = np.fliplr(clean)
            clean = np.rot90(clean, rng.integers(4))
        clean = np.ascontiguousarray(clean)

        noisy = add_gaussian_noise(clean, self.sigma, rng)
        return torch.from_numpy(noisy[None]), torch.from_numpy(clean[None])
