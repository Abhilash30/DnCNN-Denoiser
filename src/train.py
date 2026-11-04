import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.dataset import DenoiseDataset
from src.metrics import psnr
from src.model import DnCNN

torch.manual_seed(config.SEED)

train_ds = DenoiseDataset(f"{config.DATA_DIR}/train", config.PATCH_SIZE, config.SIGMA)
val_ds = DenoiseDataset(f"{config.DATA_DIR}/val", config.PATCH_SIZE, config.SIGMA, train=False)
train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

model = DnCNN()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

def validate():
    model.eval()
    scores = []
    with torch.no_grad():
        for noisy, clean in val_loader:
            out = model(noisy).clamp(0, 1)
            for o, c in zip(out, clean):
                scores.append(psnr(o[0].numpy(), c[0].numpy()))
    return np.mean(scores)

total_steps = config.EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
best = 0
for epoch in range(config.EPOCHS):
    model.train()
    start = time.time()
    total = 0
    for noisy, clean in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(noisy), clean)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    val = validate()
    print(f"epoch {epoch + 1}: train loss {total / len(train_loader):.5f}, "
          f"val {val:.2f} dB, {time.time() - start:.0f}s")
    if val > best:
        best = val
        torch.save(model.state_dict(), "best1.pt")