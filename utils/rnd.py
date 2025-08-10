from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 4, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RND(nn.Module):
    """Lightweight RND for intrinsic reward.

    - target: fixed, randomly initialized
    - predictor: trained to predict target features
    - intrinsic reward: ||predictor(obs) - target(obs)||^2
    """

    def __init__(self, in_channels: int = 4, feat_dim: int = 128, lr: float = 1e-4) -> None:
        super().__init__()
        self.target = TinyCNN(in_channels, feat_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = TinyCNN(in_channels, feat_dim)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    @torch.no_grad()
    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t = self.target(obs)
            p = self.predictor(obs)
            r = (p - t).pow(2).sum(-1)
        return r

    def update(self, obs: torch.Tensor) -> float:
        t = self.target(obs)
        p = self.predictor(obs)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 10.0)
        self.opt.step()
        return float(loss.detach().cpu())


