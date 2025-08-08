from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DreamerActionSpec:
    n: int


class DreamerAgent:
    """Minimal Dreamer-like agent skeleton.

    This class is a placeholder to wire the trainer and env. It exposes:
      - act(obs_tensor) -> action_tensor
      - update(batch_td) -> dict of logs

    For a production setup, replace internals with TorchRL Dreamer world model + loss.
    """

    def __init__(
        self,
        model_cfg,
        action_spec: Any,
        device: torch.device,
        lr: float = 3e-4,
    ) -> None:
        self.device = device

        # Infer discrete action dim from TorchRL action_spec if available
        n_actions: Optional[int] = None
        try:
            # torchrl.tensor_specs.DiscreteTensorSpec
            n_actions = int(action_spec.n)
        except Exception:
            try:
                n_actions = int(action_spec.space.n)  # gym space
            except Exception:
                pass
        if n_actions is None:
            raise RuntimeError("Could not infer discrete action dimension from action_spec")

        self.n_actions = n_actions

        # Very small CNN encoder + GRU + policy head (placeholder)
        self.encoder = nn.Sequential(
            nn.Conv2d(1 if model_cfg is None else 1, 32, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self._enc_out = None  # lazy infer
        self.gru = nn.GRU(input_size=2048, hidden_size=256, batch_first=True)
        self.actor = nn.Linear(256, self.n_actions)
        self.value = nn.Linear(256, 1)

        self.params = list(self.encoder.parameters()) + list(self.gru.parameters()) + list(self.actor.parameters()) + list(self.value.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)

        self._hidden: Optional[torch.Tensor] = None
        self.to(device)

    def to(self, device: torch.device) -> None:
        self.encoder.to(device)
        self.gru.to(device)
        self.actor.to(device)
        self.value.to(device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy policy for collection (placeholder).

        obs: tensor shaped [B, C, H, W]. Returns actions [B, 1]
        """
        self.encoder.eval()
        self.gru.eval()
        x = self.encoder(obs)
        # fabricate time dim = 1
        x = x.unsqueeze(1)
        if self._hidden is None or self._hidden.size(1) != obs.size(0):
            self._hidden = torch.zeros(1, obs.size(0), self.gru.hidden_size, device=obs.device)
        out, self._hidden = self.gru(x, self._hidden)
        logits = self.actor(out.squeeze(1))
        action = torch.argmax(logits, dim=-1, keepdim=True)
        return action

    def update(self, batch: "TensorDictBase") -> dict:  # type: ignore[name-defined]
        """Perform a single placeholder A2C-like update on a batch.

        This is NOT Dreamer; it's a scaffold to exercise the pipeline until TorchRL Dreamer
        modules are wired. Expects batch to contain keys: observation, action, reward, done.
        """
        self.encoder.train()
        self.gru.train()
        obs = batch.get("observation")  # [B, C, H, W]
        actions = batch.get("action").squeeze(-1)  # [B]
        rewards = batch.get("reward").squeeze(-1)  # [B]

        x = self.encoder(obs)
        x = x.unsqueeze(1)
        h0 = torch.zeros(1, x.size(0), self.gru.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        h = out.squeeze(1)
        logits = self.actor(h)
        values = self.value(h).squeeze(-1)

        logp = F.log_softmax(logits, dim=-1)
        logp_act = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # very rough advantage: reward - baseline
        adv = rewards.detach() - values.detach()
        policy_loss = -(logp_act * adv).mean()
        value_loss = F.mse_loss(values, rewards)
        entropy = -(logp * torch.exp(logp)).sum(-1).mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.opt.step()

        return {
            "loss/policy": float(policy_loss.detach().cpu()),
            "loss/value": float(value_loss.detach().cpu()),
            "loss/entropy": float(entropy.detach().cpu()),
        }

