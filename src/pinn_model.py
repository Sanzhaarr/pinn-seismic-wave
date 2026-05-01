import torch
import torch.nn as nn
import numpy as np


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding helps the PINN represent wave-like high-frequency behavior.
    """

    def __init__(self, input_dim=3, mapping_size=64, scale=5.0):
        super().__init__()
        B = torch.randn((input_dim, mapping_size)) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SeismicPINN(nn.Module):
    """
    PINN for approximating u(x, z, t).
    """

    def __init__(self, use_fourier=True):
        super().__init__()

        self.use_fourier = use_fourier

        if use_fourier:
            self.encoder = FourierFeatures(input_dim=3, mapping_size=64)
            input_dim = 128
        else:
            self.encoder = nn.Identity()
            input_dim = 3

        layers = []
        hidden_dim = 128
        depth = 6

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, xzt):
        features = self.encoder(xzt)
        return self.network(features)