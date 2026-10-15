"""ResNet-18 face embedder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceEmbedder(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.features(x), 1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor):
        emb = self.embed(x)
        return self.classifier(emb), emb
