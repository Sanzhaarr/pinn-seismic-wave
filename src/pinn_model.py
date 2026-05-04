import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for representing oscillatory seismic wave fields.

    The model receives normalized coordinates (x, z, t). The random projection matrix
    is stored as a non-trainable buffer so that it moves correctly between CPU/GPU
    together with the model.
    """

    def __init__(self, input_dim: int = 3, mapping_size: int = 128, scale: float = 8.0):
        super().__init__()
        projection = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("projection", projection)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * math.pi * coordinates @ self.projection
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)


class SineActivation(nn.Module):
    """
    Sine activation improves the representation of wave-like solutions compared with
    plain tanh layers, especially for high-frequency seismic responses.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SeismicPINN(nn.Module):
    """
    Physics-informed neural network for approximating the seismic displacement field
    u(x, z, t) in a heterogeneous medium.

    Expected input shape: (N, 3), where columns are normalized x, z, and t.
    Output shape: (N, 1), representing scalar displacement/pressure u.
    """

    def __init__(
        self,
        use_fourier: bool = True,
        mapping_size: int = 128,
        fourier_scale: float = 8.0,
        hidden_dim: int = 192,
        depth: int = 8,
        activation: str = "tanh",
        output_scale: float = 1.0,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError("depth must be at least 2")

        self.use_fourier = use_fourier
        self.output_scale = output_scale

        if use_fourier:
            self.encoder = FourierFeatures(
                input_dim=3,
                mapping_size=mapping_size,
                scale=fourier_scale,
            )
            input_dim = mapping_size * 2
        else:
            self.encoder = nn.Identity()
            input_dim = 3

        activation_layer = self._build_activation(activation)

        layers = [nn.Linear(input_dim, hidden_dim), activation_layer]

        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                self._build_activation(activation),
            ])

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights(activation)

    @staticmethod
    def _build_activation(activation: str) -> nn.Module:
        activation = activation.lower()

        if activation == "tanh":
            return nn.Tanh()
        if activation == "sine":
            return SineActivation()
        if activation == "gelu":
            return nn.GELU()

        raise ValueError("activation must be one of: 'tanh', 'sine', 'gelu'")

    def _initialize_weights(self, activation: str) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                if activation.lower() == "sine":
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)

    def forward(self, xzt: torch.Tensor) -> torch.Tensor:
        features = self.encoder(xzt)
        return self.output_scale * self.network(features)