from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class DreamerV3ActionSpec:
    n: int


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ELU(),
            nn.Flatten(),
        )
        # Infer flattened size lazily
        self.fc = None  # type: ignore
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        if self.fc is None:
            self.fc = nn.Linear(z.size(-1), self.latent_dim).to(z.device)
        return self.fc(z)


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int = 1, latent_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        # 4 -> 8 -> 16 -> 32 -> 64 に拡大（kernel=4, stride=2, padding=1 で厳密倍化）
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.fc(h)
        x = x.view(-1, 64, 4, 4)
        x = self.deconv(x)
        return x


class WorldModel(nn.Module):
    def __init__(self, obs_channels: int = 1, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = ConvEncoder(obs_channels, latent_dim)
        self.rssm = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.decoder = ConvDecoder(obs_channels, latent_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs_seq: [B, T, C, H, W]
        B, T = obs_seq.size(0), obs_seq.size(1)
        enc = []
        for t in range(T):
            enc.append(self.encoder(obs_seq[:, t]))
        enc = torch.stack(enc, dim=1)  # [B,T,latent]
        h0 = torch.zeros(1, B, enc.size(-1), device=enc.device)
        h_seq, _ = self.rssm(enc, h0)  # [B,T,latent]
        # reconstruct each step from h
        recons = []
        rews = []
        for t in range(T):
            h = h_seq[:, t]
            recons.append(self.decoder(h))
            rews.append(self.reward_head(h))
        recon = torch.stack(recons, dim=1)  # [B,T,C,H,W]
        reward_pred = torch.stack(rews, dim=1).squeeze(-1)  # [B,T]
        return h_seq, recon, reward_pred


class ActorCritic(nn.Module):
    def __init__(self, latent_dim: int, n_actions: int, llm_features_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.llm_features_dim = llm_features_dim
        
        # Feature fusion for LLM features
        if llm_features_dim > 0:
            self.feature_norm = nn.LayerNorm(llm_features_dim)
            self.feature_proj = nn.Linear(llm_features_dim, latent_dim // 4)
            actor_input_dim = latent_dim + latent_dim // 4
        else:
            self.feature_norm = None
            self.feature_proj = None
            actor_input_dim = latent_dim
        
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def policy_logits(self, h: torch.Tensor, llm_features: Optional[torch.Tensor] = None,
                     llm_logits: Optional[torch.Tensor] = None, 
                     llm_mask: Optional[torch.Tensor] = None,
                     alpha: float = 0.0) -> torch.Tensor:
        """Compute policy logits with optional LLM integration."""
        # Feature concatenation (late fusion)
        actor_input = h
        if self.feature_proj is not None:
            if llm_features is not None:
                llm_feat_norm = self.feature_norm(llm_features)
                llm_feat_proj = self.feature_proj(llm_feat_norm)
            else:
                # LLM特徴が未提供でも形状を合わせるためにゼロを連結
                llm_feat_proj = torch.zeros(
                    h.size(0), self.feature_proj.out_features, device=h.device, dtype=h.dtype
                )
            actor_input = torch.cat([h, llm_feat_proj], dim=-1)
        
        # Base actor logits
        logits = self.actor(actor_input)
        
        # LLM prior mixing
        if llm_logits is not None and alpha > 0.0:
            if llm_logits.shape == logits.shape:
                logits = logits + alpha * llm_logits
        
        # Action masking
        if llm_mask is not None:
            if llm_mask.shape == logits.shape:
                mask_penalty = torch.where(llm_mask == 0, -1e9, 0.0)
                logits = logits + mask_penalty
        
        return logits

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return self.critic(h).squeeze(-1)


class DreamerV3Agent:
    """Minimal DreamerV3-like agent (PyTorch) suitable for Crafter 64x64 grayscale.

    This is a compact approximation: encoder+RSSM+decoder+reward; actor-critic in latent space.
    It supports optional KL regularization to an external LLM prior (discrete actions).
    """

    def __init__(
        self,
        model_cfg: Any,
        action_spec: Any,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.05,
        epsilon_greedy: float = 0.0,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.lambda_kl: float = 0.0
        self.entropy_coef = float(entropy_coef)
        self.epsilon_greedy = float(epsilon_greedy)

        # Infer discrete action dim
        n_actions: Optional[int] = None
        try:
            n_actions = int(action_spec.n)
        except Exception:
            try:
                n_actions = int(action_spec.space.n)
            except Exception:
                pass
        if n_actions is None:
            raise RuntimeError("Could not infer discrete action dimension from action_spec")
        self.n_actions = n_actions

        # 入力チャンネル数（GrayScale+FrameStack の場合は frame_stack 個）。
        # env 側の前処理に依存するため、cfg.model.obs_channels を優先し、無ければ 1。
        obs_channels = int(getattr(model_cfg, "obs_channels", 1))
        latent_dim = int(getattr(model_cfg, "latent_dim", 256))
        llm_features_dim = int(getattr(model_cfg, "llm_features_dim", 0))

        self.world = WorldModel(obs_channels=obs_channels, latent_dim=latent_dim)
        self.ac = ActorCritic(latent_dim=latent_dim, n_actions=self.n_actions, 
                             llm_features_dim=llm_features_dim)
        
        # LLM integration parameters
        self.alpha_schedule = AlphaScheduler(initial_alpha=0.1, decay_rate=0.999)
        self.use_llm_features = llm_features_dim > 0

        self.params = list(self.world.parameters()) + list(self.ac.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self._enc_cache = ConvEncoder(in_channels=obs_channels, latent_dim=latent_dim)  # for single-step act
        self.to(device)

    def to(self, device: torch.device) -> None:
        self.world.to(device)
        self.ac.to(device)
        self._enc_cache.to(device)

    def save(self, path: str) -> None:
        state = {
            "world": self.world.state_dict(),
            "ac": self.ac.state_dict(),
            "opt": self.opt.state_dict(),
            "lambda_kl": self.lambda_kl,
            "gamma": self.gamma,
            "n_actions": self.n_actions,
        }
        torch.save(state, path)

    def load(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        self.world.load_state_dict(state["world"], strict=strict)
        self.ac.load_state_dict(state["ac"], strict=strict)
        try:
            self.opt.load_state_dict(state["opt"])  # optimizer は任意
        except Exception:
            pass
        self.lambda_kl = float(state.get("lambda_kl", self.lambda_kl))
        self.gamma = float(state.get("gamma", self.gamma))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B,C,H,W]
        self.world.eval()
        self.ac.eval()
        z = self._enc_cache(obs)
        h, _ = self.world.rssm(z.unsqueeze(1), torch.zeros(1, obs.size(0), z.size(-1), device=obs.device))
        h = h.squeeze(1)
        logits = self.ac.policy_logits(h)
        # ε-greedy: 少確率でランダム行動を混入
        if self.epsilon_greedy > 0.0:
            import torch as _torch
            B = logits.size(0)
            n = logits.size(-1)
            greedy = torch.argmax(logits, dim=-1)
            rnd = _torch.randint(low=0, high=n, size=(B,), device=logits.device)
            mask = (_torch.rand(B, device=logits.device) < self.epsilon_greedy)
            action = torch.where(mask, rnd, greedy).unsqueeze(-1)
        else:
            action = torch.argmax(logits, dim=-1, keepdim=True)
        return action

    def _td0_target(self, rewards: torch.Tensor, values_next: torch.Tensor, dones: Optional[torch.Tensor]) -> torch.Tensor:
        """TD(0) 目標: r_t + gamma * (1-done_t) * V(s_{t+1})

        ここでは T=1 前提のため簡略化。
        """
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if values_next.dim() == 1:
            values_next = values_next.unsqueeze(-1)
        if dones is None:
            mask = torch.ones_like(rewards)
        else:
            mask = (1.0 - dones.float())
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            elif mask.dim() > 2:
                mask = mask.squeeze(-1)
        return rewards + self.gamma * mask * values_next

    def update(self, batch: "TensorDictBase") -> dict:  # type: ignore[name-defined]
        self.world.train()
        self.ac.train()

        # batch keys: observation [B,C,H,W], action [B,1], reward [B,1]; we reshape to sequences of T=1
        obs = batch.get("observation").to(self.device)
        actions = batch.get("action").to(self.device)
        # 形状 [B] に正規化し、ロング型へ
        if actions.dtype != torch.long:
            actions = actions.to(torch.long)
        while actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        rewards = batch.get("reward").to(self.device).squeeze(-1)
        dones = batch.get("done").to(self.device).squeeze(-1) if "done" in batch.keys(True, True) else None

        # Use a pseudo-sequence of length 1 for world model consistency
        obs_seq = obs.unsqueeze(1)  # [B,1,C,H,W]
        h_seq, recon_seq, reward_pred_seq = self.world(obs_seq)
        h = h_seq[:, 0]
        recon = recon_seq[:, 0]
        reward_pred = reward_pred_seq[:, 0]

        # Reconstruction loss (likelihood surrogate)
        recon_loss = F.mse_loss(recon, obs)
        # Reward prediction loss
        reward_loss = F.mse_loss(reward_pred, rewards)

        # Extract LLM data if available
        llm_features = None
        llm_logits = None
        llm_mask = None
        current_alpha = 0.0
        
        if "llm_features" in batch.keys(True, True) and self.use_llm_features:
            llm_features = batch.get("llm_features").to(self.device)
            if llm_features.numel() > 0:  # Check if not empty
                # Normalize features to [-1, 1] range
                llm_features = torch.clamp(llm_features, -3.0, 3.0) / 3.0
        
        if "llm_prior_logits" in batch.keys(True, True):
            llm_logits = batch.get("llm_prior_logits").to(self.device)
            if "llm_confidence" in batch.keys(True, True) and "llm_mask" in batch.keys(True, True):
                conf = batch.get("llm_confidence").to(self.device).squeeze(-1)
                mask = batch.get("llm_mask").to(self.device).squeeze(-1)
                # Adaptive alpha based on confidence and mask
                current_alpha = self.alpha_schedule.get_alpha() * conf * mask
                current_alpha = current_alpha.mean().item()  # Scalar for this batch
        
        if "llm_mask" in batch.keys(True, True):
            # Convert from RBExtra bitmask format if needed
            mask_bits = batch.get("llm_mask").to(self.device)
            if mask_bits.dim() == 1:  # Bitmask format
                llm_mask = self._bitmask_to_tensor(mask_bits, self.n_actions)
        
        # Policy and value on latent with LLM integration
        logits = self.ac.policy_logits(h, llm_features, llm_logits, llm_mask, current_alpha)
        values = self.ac.value(h).unsqueeze(-1)  # [B,1] 現時点の V(s_t)
        
        # Update alpha schedule
        self.alpha_schedule.step()

        logp = F.log_softmax(logits, dim=-1)
        logp_act = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # 単ステップの TD(0)
        with torch.no_grad():
            target_v = self._td0_target(rewards, values.detach(), dones).squeeze(-1)
        advantage = (target_v - values.squeeze(-1)).detach()

        policy_loss = -(logp_act * advantage).mean()
        value_loss = F.mse_loss(values.squeeze(-1), target_v)

        # Optional KL to LLM prior
        kl_term = torch.tensor(0.0, device=obs.device)
        if "llm_prior_logits" in batch.keys(True, True):
            try:
                from torch.distributions import Categorical

                pi_dist = Categorical(logits=logits)
                rho_dist = Categorical(logits=batch.get("llm_prior_logits").to(obs.device))
                kl = torch.distributions.kl.kl_divergence(pi_dist, rho_dist)
                if "llm_confidence" in batch.keys(True, True) and "llm_mask" in batch.keys(True, True):
                    w = (
                        batch.get("llm_confidence").to(obs.device).squeeze(-1)
                        * batch.get("llm_mask").to(obs.device).squeeze(-1)
                    ).detach()
                    kl = kl * w
                kl_term = kl.mean()
            except Exception:
                kl_term = torch.tensor(0.0, device=obs.device)

        entropy = -(logp * torch.exp(logp)).sum(-1).mean()

        loss_model = recon_loss + reward_loss
        loss_ac = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy + self.lambda_kl * kl_term
        loss = loss_model + loss_ac

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.opt.step()

        return {
            "loss/model_recon": float(recon_loss.detach().cpu()),
            "loss/model_reward": float(reward_loss.detach().cpu()),
            "loss/policy": float(policy_loss.detach().cpu()),
            "loss/value": float(value_loss.detach().cpu()),
            "loss/entropy": float(entropy.detach().cpu()),
            "loss/kl": float(kl_term.detach().cpu()),
            "llm/alpha": current_alpha,
            "llm/used_features": float(llm_features is not None),
            "llm/used_logits": float(llm_logits is not None),
        }
    
    def _bitmask_to_tensor(self, mask_bits: torch.Tensor, n_actions: int) -> torch.Tensor:
        """Convert bitmask to action mask tensor."""
        batch_size = mask_bits.size(0)
        mask = torch.zeros(batch_size, n_actions, device=mask_bits.device)
        
        for i in range(min(n_actions, 64)):  # Support up to 64 actions
            bit_mask = (mask_bits & (1 << i)) != 0
            mask[:, i] = bit_mask.float()
        
        return mask


class AlphaScheduler:
    """Schedule alpha parameter for LLM integration."""
    
    def __init__(self, initial_alpha: float = 0.1, decay_rate: float = 0.999, 
                 min_alpha: float = 0.01):
        self.initial_alpha = initial_alpha
        self.current_alpha = initial_alpha
        self.decay_rate = decay_rate
        self.min_alpha = min_alpha
        self.step_count = 0
    
    def get_alpha(self) -> float:
        return max(self.min_alpha, self.current_alpha)
    
    def step(self) -> None:
        self.step_count += 1
        self.current_alpha = max(
            self.min_alpha, 
            self.initial_alpha * (self.decay_rate ** self.step_count)
        )
    
    def reset(self) -> None:
        self.current_alpha = self.initial_alpha
        self.step_count = 0


