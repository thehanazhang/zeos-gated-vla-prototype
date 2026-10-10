"""Small conditional rectified-flow transformer; only this module is trained."""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class HeadConfig:
    feature_dim: int = 2560
    proprio_dim: int = 7
    action_dim: int = 7
    horizon: int = 16
    width: int = 128
    layers: int = 2
    heads: int = 4


class FlowHead(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        if config.horizon != 16 or config.width % config.heads:
            raise ValueError("Expected 16 steps and width divisible by heads")
        self.config = config
        self.feature = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.width),
        )
        self.proprio = nn.Linear(config.proprio_dim, config.width)
        self.action = nn.Linear(config.action_dim, config.width)
        self.time = nn.Sequential(
            nn.Linear(32, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.position = nn.Parameter(
            torch.randn(1, config.horizon, config.width) * 0.02
        )
        layer = nn.TransformerDecoderLayer(
            config.width,
            config.heads,
            config.width * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(
            layer, config.layers, norm=nn.LayerNorm(config.width)
        )
        self.out = nn.Linear(config.width, config.action_dim)
        self.register_buffer(
            "frequencies", torch.exp(torch.linspace(0, math.log(1000), 16))
        )

    def forward(self, x, t, features, proprio, feature_mask):
        c = self.config
        if x.ndim != 3 or tuple(x.shape[1:]) != (16, c.action_dim):
            raise ValueError("Action shape must be [batch,16,action_dim]")
        if features.ndim != 3 or features.shape[-1] != c.feature_dim:
            raise ValueError("Feature dimension mismatch")
        if proprio.shape != (x.shape[0], c.proprio_dim) or t.shape != (x.shape[0],):
            raise ValueError("Proprioception or time shape mismatch")
        if (
            feature_mask.shape != features.shape[:2]
            or not feature_mask.bool().any(-1).all()
        ):
            raise ValueError("Every sample must have at least one valid feature token")
        phase = t[:, None] * self.frequencies[None, :] * 2 * math.pi
        emb = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        tokens = (
            self.action(x)
            + self.position
            + emb[:, None]
            + self.proprio(proprio)[:, None]
        )
        memory = self.feature(features.to(x.dtype))
        return self.out(
            self.transformer(
                tokens, memory, memory_key_padding_mask=~feature_mask.bool()
            )
        )

    def loss(
        self, actions, features, proprio, feature_mask, action_mask=None, generator=None
    ):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite recorded actions")
        noise = torch.randn(
            actions.shape,
            device=actions.device,
            dtype=actions.dtype,
            generator=generator,
        )
        t = torch.rand(actions.shape[0], device=actions.device, generator=generator)
        x = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        error = (
            self(x, t, features, proprio, feature_mask) - (actions - noise)
        ).square()
        mask = (
            torch.ones_like(actions[..., 0], dtype=torch.bool)
            if action_mask is None
            else action_mask.bool()
        )
        if mask.shape != actions.shape[:2] or not mask.any():
            raise ValueError("Empty or invalid action mask")
        return (error * mask[..., None]).sum() / (mask.sum() * actions.shape[-1])

    @torch.no_grad()
    def sample(self, features, proprio, feature_mask, steps=32, generator=None):
        if steps < 1:
            raise ValueError("Euler steps must be positive")
        batch = proprio.shape[0]
        x = torch.randn(
            (batch, 16, self.config.action_dim),
            device=proprio.device,
            dtype=proprio.dtype,
            generator=generator,
        )
        for i in range(steps):
            t = torch.full((batch,), i / steps, device=x.device, dtype=x.dtype)
            x = x + self(x, t, features, proprio, feature_mask) / steps
        if not torch.isfinite(x).all():
            raise RuntimeError("Nonfinite policy output")
        return x

    def specification(self):
        return asdict(self.config)
