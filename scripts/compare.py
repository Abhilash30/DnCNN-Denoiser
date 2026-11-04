import argparse

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

import config
from src.metrics import psnr
from src.model import DnCNN
from src.noise import add_gaussian_noise

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--add-noise", action="store_true")
parser.add_argument("--blur", type=float, default=1.0)
args = parser.parse_args()

img = np.asarray(Image.open(args.image).convert("L"), dtype=np.float32) / 255.0

if args.add_noise:
    clean = img
    noisy = add_gaussian_noise(clean, config.SIGMA, np.random.default_rng(config.SEED))
else:
    clean = None
    noisy = img

model = DnCNN()
model.load_state_dict(torch.load("models/best1.pt"))
model.eval()
with torch.no_grad():
    denoised = model(torch.from_numpy(noisy[None, None])).clamp(0, 1)[0, 0].numpy()

blur = args.blur
if clean is not None:
    blur = max(np.arange(0.5, 3.01, 0.25), key=lambda s: psnr(gaussian_filter(noisy, s), clean))
blurred = gaussian_filter(noisy, blur)

if clean is not None:
    print(f"noisy:    {psnr(noisy, clean):.2f} dB")
    print(f"gaussian: {psnr(blurred, clean):.2f} dB (best blur sigma = {blur:.2f})")
    print(f"model:    {psnr(denoised, clean):.2f} dB")


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


panels = [noisy, blurred, denoised]
Image.fromarray(np.concatenate([to_u8(p) for p in panels], axis=1)).save("comparison.png")