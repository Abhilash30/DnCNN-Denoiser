import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import config 

from src.dataset import DenoiseDataset

ds = DenoiseDataset(f"{config.DATA_DIR}/train", config.PATCH_SIZE, config.SIGMA)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

noisy, clean = next(iter(loader))
print(noisy.shape, clean.shape)


def to_row(batch):
    return np.concatenate([(x[0].numpy() * 255).astype(np.uint8) for x in batch[:8]], axis=1)


grid = np.concatenate([to_row(noisy), to_row(clean)], axis=0)
Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save("check.png")