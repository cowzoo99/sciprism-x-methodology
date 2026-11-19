from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class StateFieldModel(nn.Module):
    def __init__(
        self,
        *,
        dimensions: tuple[int, int, int] = (15, 64, 16),
        energy_dimensions: tuple[int, int, int] = (31, 32, 1),
        activation: str = "gelu",
        gelu_approximate: str = "none",
        linear_bias: bool = True,
        initialization: str = "kaiming_uniform_linear_default",
        dropout: float = 0.0,
        dropout_inplace: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if activation != "gelu":
            raise ValueError("state field currently requires the paper GELU activation")
        if gelu_approximate != "none":
            raise ValueError("state field GELU approximation must be explicit 'none'")
        if linear_bias is not True or dropout_inplace is not False:
            raise ValueError("state field bias/dropout controls differ from the method contract")
        if initialization != "kaiming_uniform_linear_default":
            raise ValueError("unsupported state field initialization")
        if dimensions[0] != 15 or energy_dimensions[0] != dimensions[0] + dimensions[2]:
            raise ValueError("state and energy dimensions do not match the 15-dimensional field")
        if energy_dimensions[2] != 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("energy output must be scalar and dropout must be in [0, 1)")
        constructor_start_state = torch.get_rng_state()
        encoder_layers: list[nn.Module] = [
            nn.Linear(dimensions[0], dimensions[1], bias=linear_bias),
            nn.GELU(approximate=gelu_approximate),
        ]
        if dropout > 0.0:
            encoder_layers.append(nn.Dropout(dropout, inplace=dropout_inplace))
        encoder_layers.append(
            nn.Linear(dimensions[1], dimensions[2], bias=linear_bias)
        )
        self.encoder = nn.Sequential(
            *encoder_layers,
        )
        self.regime_head = nn.Linear(dimensions[2], 3, bias=linear_bias)
        self.energy_head = nn.Sequential(
            nn.Linear(
                energy_dimensions[0],
                energy_dimensions[1],
                bias=linear_bias,
            ),
            nn.GELU(approximate=gelu_approximate),
            nn.Linear(
                energy_dimensions[1],
                energy_dimensions[2],
                bias=linear_bias,
            ),
        )
        # Rewind the constructor RNG consumption, then reproduce nn.Linear's
        # initialization explicitly so the algorithm is version-auditable
        # without changing the established seeded parameter realization.
        torch.set_rng_state(constructor_start_state)
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            nn.init.kaiming_uniform_(
                module.weight,
                a=math.sqrt(5.0),
                mode="fan_in",
                nonlinearity="leaky_relu",
            )
            if module.bias is not None:
                fan_in = int(module.weight.shape[1])
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(module.bias, -bound, bound)
        self.to(dtype=dtype)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.encoder(state)
        regime_logits = self.regime_head(embedding)
        energy = self.energy_head(torch.cat([embedding, state], dim=-1))
        return regime_logits, energy
