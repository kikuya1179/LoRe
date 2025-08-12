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
        # Ensure channel count consistency by lightweight projection if needed
        if obs.dim() == 4 and obs.size(1) != next(self.predictor.net[0].parameters()).size(1):
            # Simple fix: if channels mismatch, average or tile to match expected channels
            c_in = obs.size(1)
            c_exp = int(next(self.predictor.net[0].parameters()).size(1))
            if c_in > c_exp:
                obs = obs[:, :c_exp]
            elif c_in < c_exp:
                rep = (c_exp + c_in - 1) // c_in
                obs = obs.repeat(1, rep, 1, 1)[:, :c_exp]
        t = self.target(obs)
        p = self.predictor(obs)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 10.0)
        self.opt.step()
        return float(loss.detach().cpu())




