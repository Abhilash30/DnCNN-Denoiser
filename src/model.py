import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, depth=17, width=64):
        super().__init__()
        layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(width, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)