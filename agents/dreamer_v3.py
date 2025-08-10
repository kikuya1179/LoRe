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

    def policy_logits(self, h: torch.Tensor, values: Optional[torch.Tensor] = None,
                     llm_features: Optional[torch.Tensor] = None,
                     llm_logits: Optional[torch.Tensor] = None, 
                     llm_mask: Optional[torch.Tensor] = None,
                     model_error: Optional[torch.Tensor] = None,
                     return_uncertainty: bool = False) -> torch.Tensor:
        """Compute policy logits with uncertainty-based LLM integration following LoRe spec."""
        # Feature concatenation (late fusion)
        actor_input = h
        if self.feature_proj is not None:
            if llm_features is not None:
                llm_feat_norm = self.feature_norm(llm_features)
                llm_feat_proj = self.feature_proj(llm_feat_norm)
            else:
                llm_feat_proj = torch.zeros(
                    h.size(0), self.feature_proj.out_features, device=h.device, dtype=h.dtype
                )
            actor_input = torch.cat([h, llm_feat_proj], dim=-1)
        
        # Base world model policy logits
        logits_wm = self.actor(actor_input)
        
        # If no LLM integration, return base logits
        if llm_logits is None:
            if return_uncertainty:
                # Access uncertainty_gate from parent agent if available
                if hasattr(self, '_agent_ref') and self._agent_ref is not None:
                    uncertainty = self._agent_ref.uncertainty_gate.compute_uncertainty(
                        logits_wm, values if values is not None else torch.zeros_like(logits_wm[:, :1]),
                        model_error
                    )
                else:
                    uncertainty = torch.zeros(logits_wm.size(0), device=logits_wm.device)
                return logits_wm, uncertainty
            return logits_wm
        
        # Compute uncertainty for β gating  
        if values is None:
            values = torch.zeros(h.size(0), 1, device=h.device)  # Dummy values if not provided
        
        # Access uncertainty_gate from parent agent
        if hasattr(self, '_agent_ref') and self._agent_ref is not None:
            uncertainty = self._agent_ref.uncertainty_gate.compute_uncertainty(logits_wm, values, model_error)
            beta = self._agent_ref.uncertainty_gate.compute_beta(uncertainty)
            
            # Store beta for monitoring
            self._agent_ref.uncertainty_gate.beta_history.extend(beta.detach().cpu().tolist())
            if len(self._agent_ref.uncertainty_gate.beta_history) > 1000:
                self._agent_ref.uncertainty_gate.beta_history = self._agent_ref.uncertainty_gate.beta_history[-500:]
        else:
            # Fallback if agent reference not available
            uncertainty = torch.zeros(h.size(0), device=h.device)
            beta = torch.zeros(h.size(0), device=h.device)
        
        # LoRe Policy Prior: logits_mix = logits_wm + β(s) * stopgrad(logits_llm)
        if llm_logits.shape == logits_wm.shape:
            # Critical: stopgrad on LLM logits to prevent gradient flow
            llm_logits_sg = llm_logits.detach()
            
            # Apply β gating with broadcasting
            if beta.dim() == 1 and logits_wm.dim() == 2:
                beta_expanded = beta.unsqueeze(-1).expand_as(logits_wm)
            else:
                beta_expanded = beta
            
            logits_mix = logits_wm + beta_expanded * llm_logits_sg
        else:
            logits_mix = logits_wm  # Fallback if shapes don't match
        
        # Action masking (applied after mixing)
        if llm_mask is not None:
            if llm_mask.shape == logits_mix.shape:
                mask_penalty = torch.where(llm_mask == 0, -1e9, 0.0)
                logits_mix = logits_mix + mask_penalty
        
        if return_uncertainty:
            return logits_mix, uncertainty, beta
        return logits_mix

    def value(self, h: torch.Tensor, return_ensemble: bool = False) -> torch.Tensor:
        """Compute value with optional ensemble for uncertainty estimation."""
        base_value = self.critic(h).squeeze(-1)
        
        # Access critic_ensemble from agent reference if available
        critic_ensemble = None
        if hasattr(self, '_agent_ref') and self._agent_ref is not None:
            critic_ensemble = getattr(self._agent_ref, 'critic_ensemble', None)
        
        if not return_ensemble or critic_ensemble is None:
            return base_value
        
        # Compute ensemble values for uncertainty estimation
        ensemble_values = []
        for critic in critic_ensemble:
            ensemble_values.append(critic(h).squeeze(-1))
        ensemble_values = torch.stack(ensemble_values, dim=0)  # [num_critics, batch_size]
        
        return base_value, ensemble_values


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
        self.lambda_bc: float = getattr(model_cfg, "lambda_bc", 0.1)  # BC regularization for synthetic data
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
        
        # Set agent reference for uncertainty gate access
        self.ac._agent_ref = self
        
        # Enhanced LLM integration with uncertainty gating
        self.uncertainty_gate = UncertaintyGate(
            beta_max=getattr(model_cfg, "beta_max", 0.3),
            delta_target=getattr(model_cfg, "delta_target", 0.1),
            kl_lr=getattr(model_cfg, "kl_lr", 1e-3),
            uncertainty_threshold=getattr(model_cfg, "uncertainty_threshold", 0.5)
        )
        self.use_llm_features = llm_features_dim > 0
        
        # Value ensemble for uncertainty estimation (optional)
        self.use_value_ensemble = getattr(model_cfg, "use_value_ensemble", False)
        if self.use_value_ensemble:
            self.critic_ensemble = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ELU(),
                    nn.Linear(256, 1),
                ) for _ in range(3)
            ])
        else:
            self.critic_ensemble = None

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

        # Extract LLM data and compute enhanced integration
        llm_features = None
        llm_logits = None
        llm_mask = None
        uncertainty = None
        beta_values = None
        
        if "llm_features" in batch.keys(True, True) and self.use_llm_features:
            llm_features = batch.get("llm_features").to(self.device)
            if llm_features.numel() > 0:  # Check if not empty
                # Normalize features to [-1, 1] range
                llm_features = torch.clamp(llm_features, -3.0, 3.0) / 3.0
        
        if "llm_prior_logits" in batch.keys(True, True):
            llm_logits = batch.get("llm_prior_logits").to(self.device)
        
        if "llm_mask" in batch.keys(True, True):
            # Convert from RBExtra bitmask format if needed
            mask_bits = batch.get("llm_mask").to(self.device)
            if mask_bits.dim() == 1:  # Bitmask format
                llm_mask = self._bitmask_to_tensor(mask_bits, self.n_actions)
        
        # Compute values (with ensemble if available)
        if self.critic_ensemble is not None:
            values, ensemble_values = self.ac.value(h, return_ensemble=True)
            value_variance = torch.var(ensemble_values, dim=0)  # [batch_size]
        else:
            values = self.ac.value(h)
            value_variance = None
        
        values = values.unsqueeze(-1)  # [B,1]
        
        # Policy logits with uncertainty-based LLM integration
        logits_result = self.ac.policy_logits(
            h, values, llm_features, llm_logits, llm_mask, 
            model_error=value_variance, return_uncertainty=True
        )
        
        if isinstance(logits_result, tuple):
            if len(logits_result) == 3:
                logits, uncertainty, beta_values = logits_result
            else:
                logits, uncertainty = logits_result
                beta_values = None
        else:
            logits = logits_result
            uncertainty = None
            beta_values = None

        logp = F.log_softmax(logits, dim=-1)
        logp_act = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # 単ステップの TD(0)
        with torch.no_grad():
            target_v = self._td0_target(rewards, values.detach(), dones).squeeze(-1)
        advantage = (target_v - values.squeeze(-1)).detach()

        policy_loss = -(logp_act * advantage).mean()
        value_loss = F.mse_loss(values.squeeze(-1), target_v)

        # Enhanced KL control following LoRe specification
        kl_divergence = torch.tensor(0.0, device=obs.device)
        kl_penalty = torch.tensor(0.0, device=obs.device)
        
        if llm_logits is not None:
            try:
                from torch.distributions import Categorical
                
                # Compute KL divergence between mixed policy and base policy
                pi_mixed = Categorical(logits=logits)
                pi_base = Categorical(logits=self.ac.policy_logits(h, values, llm_features=llm_features))
                kl_divergence = torch.distributions.kl.kl_divergence(pi_mixed, pi_base)
                kl_mean = kl_divergence.mean()
                
                # Update uncertainty gate's KL constraint
                self.uncertainty_gate.update_kl_constraint(kl_mean)
                
                # Compute KL penalty using Lagrange multiplier
                kl_penalty_coeff = self.uncertainty_gate.get_kl_penalty_coeff()
                kl_penalty = kl_penalty_coeff * torch.relu(kl_mean - self.uncertainty_gate.delta_target)
                
            except Exception:
                kl_divergence = torch.tensor(0.0, device=obs.device)
                kl_penalty = torch.tensor(0.0, device=obs.device)

        entropy = -(logp * torch.exp(logp)).sum(-1).mean()

        loss_model = recon_loss + reward_loss
        # Behavioral Cloning regularization for synthetic data (LoRe specification)
        bc_loss = torch.tensor(0.0, device=obs.device)
        
        if "_is_synthetic" in batch.keys(True, True):
            is_synthetic = batch.get("_is_synthetic").to(obs.device)
            synthetic_mask = is_synthetic.float()
            
            if synthetic_mask.sum() > 0 and llm_logits is not None:
                # BC loss: encourage policy to match LLM actions on synthetic data
                # L_BC = λ_bc * E_synth[-log π_ψ(a_t | s_t)]
                log_probs_synthetic = logp * synthetic_mask.unsqueeze(-1)
                bc_loss = -log_probs_synthetic.sum(dim=-1).mean()
                bc_loss = self.lambda_bc * bc_loss * (synthetic_mask.sum() / synthetic_mask.numel())
        
        # Enhanced loss with BC regularization
        loss_ac = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy + kl_penalty + bc_loss
        loss = loss_model + loss_ac

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.opt.step()

        # Collect enhanced metrics
        metrics = {
            "loss/model_recon": float(recon_loss.detach().cpu()),
            "loss/model_reward": float(reward_loss.detach().cpu()),
            "loss/policy": float(policy_loss.detach().cpu()),
            "loss/value": float(value_loss.detach().cpu()),
            "loss/entropy": float(entropy.detach().cpu()),
            "loss/kl_divergence": float(kl_divergence.mean().detach().cpu()) if kl_divergence.numel() > 1 else float(kl_divergence.detach().cpu()),
            "loss/kl_penalty": float(kl_penalty.detach().cpu()),
            "loss/bc_regularization": float(bc_loss.detach().cpu()),
            "llm/used_features": float(llm_features is not None),
            "llm/used_logits": float(llm_logits is not None),
        }
        
        # Add synthetic data metrics if available
        if "_is_synthetic" in batch.keys(True, True):
            is_synthetic = batch.get("_is_synthetic").to(obs.device)
            synthetic_ratio = float(is_synthetic.float().mean().detach().cpu())
            metrics["synthetic/ratio"] = synthetic_ratio
            
            if "_sample_weights" in batch.keys(True, True):
                sample_weights = batch.get("_sample_weights").to(obs.device)
                synthetic_weights = sample_weights[is_synthetic]
                if synthetic_weights.numel() > 0:
                    metrics["synthetic/avg_weight"] = float(synthetic_weights.mean().detach().cpu())
                    metrics["synthetic/weight_std"] = float(synthetic_weights.std().detach().cpu())
        
        # Add uncertainty gate metrics
        if uncertainty is not None:
            metrics["uncertainty/mean"] = float(uncertainty.mean().detach().cpu())
            metrics["uncertainty/std"] = float(uncertainty.std().detach().cpu())
        if beta_values is not None:
            metrics["beta/mean"] = float(beta_values.mean().detach().cpu())
            metrics["beta/std"] = float(beta_values.std().detach().cpu())
        
        # Add uncertainty gate internal metrics
        gate_metrics = self.uncertainty_gate.get_metrics()
        metrics.update(gate_metrics)
        
        return metrics
    
    def _bitmask_to_tensor(self, mask_bits: torch.Tensor, n_actions: int) -> torch.Tensor:
        """Convert bitmask to action mask tensor."""
        batch_size = mask_bits.size(0)
        mask = torch.zeros(batch_size, n_actions, device=mask_bits.device)
        
        for i in range(min(n_actions, 64)):  # Support up to 64 actions
            bit_mask = (mask_bits & (1 << i)) != 0
            mask[:, i] = bit_mask.float()
        
        return mask


class UncertaintyGate:
    """Uncertainty-based β gating for LLM integration following LoRe specification."""
    
    def __init__(self, beta_max: float = 0.3, delta_target: float = 0.1, 
                 kl_lr: float = 1e-3, uncertainty_threshold: float = 0.5):
        self.beta_max = beta_max
        self.delta_target = delta_target
        self.kl_lr = kl_lr
        self.uncertainty_threshold = uncertainty_threshold
        
        # Lagrange multiplier for target KL constraint
        self.lambda_kl = torch.tensor(0.01, dtype=torch.float32)
        
        # Running statistics for uncertainty normalization
        self.entropy_ema = torch.tensor(0.5, dtype=torch.float32)
        self.value_var_ema = torch.tensor(0.1, dtype=torch.float32)
        self.model_error_ema = torch.tensor(0.1, dtype=torch.float32)
        self.ema_decay = 0.99
        
        # Metrics tracking
        self.kl_history = []
        self.beta_history = []
        
    def compute_uncertainty(self, logits: torch.Tensor, values: torch.Tensor, 
                          model_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute uncertainty score from multiple sources."""
        batch_size = logits.size(0)
        device = logits.device
        
        # 1. Policy entropy (high entropy = high uncertainty)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        entropy_normalized = entropy / (entropy.max() + 1e-8)
        
        # 2. Value variance (estimate with small noise perturbation)
        if values.dim() > 1:
            values = values.squeeze(-1)
        noise = torch.randn_like(values) * 0.01
        # Use population variance (correction=0) to avoid NaN when batch size == 1
        try:
            value_var = torch.var(values + noise, dim=0, correction=0, keepdim=True)
        except TypeError:
            # Older PyTorch fallback (unbiased=False)
            value_var = torch.var(values + noise, dim=0, keepdim=True, unbiased=False)
        value_var = torch.nan_to_num(value_var, nan=0.0, posinf=0.0, neginf=0.0)
        value_var = value_var.expand(batch_size)  # Broadcast to batch
        
        # 3. Model disagreement/error (if available)
        if model_error is not None:
            model_uncertainty = model_error.squeeze(-1) if model_error.dim() > 1 else model_error
            if model_uncertainty.numel() == 1:
                model_uncertainty = model_uncertainty.expand(batch_size)
        else:
            model_uncertainty = torch.zeros(batch_size, device=device)
        model_uncertainty = torch.nan_to_num(model_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update EMAs for normalization (no grad; detach source means)
        with torch.no_grad():
            ent_mean = torch.nan_to_num(entropy.mean().detach(), nan=0.0, posinf=0.0, neginf=0.0)
            val_mean = torch.nan_to_num(value_var.mean().detach(), nan=0.0, posinf=0.0, neginf=0.0)
            self.entropy_ema = torch.nan_to_num(
                self.ema_decay * self.entropy_ema + (1 - self.ema_decay) * ent_mean,
                nan=self.entropy_ema.item(), posinf=self.entropy_ema.item(), neginf=self.entropy_ema.item()
            )
            self.value_var_ema = torch.nan_to_num(
                self.ema_decay * self.value_var_ema + (1 - self.ema_decay) * val_mean,
                nan=self.value_var_ema.item(), posinf=self.value_var_ema.item(), neginf=self.value_var_ema.item()
            )
            if model_error is not None:
                mod_mean = torch.nan_to_num(model_uncertainty.mean().detach(), nan=0.0, posinf=0.0, neginf=0.0)
                self.model_error_ema = torch.nan_to_num(
                    self.ema_decay * self.model_error_ema + (1 - self.ema_decay) * mod_mean,
                    nan=self.model_error_ema.item(), posinf=self.model_error_ema.item(), neginf=self.model_error_ema.item()
                )
        
        # Normalize and combine uncertainty sources
        entropy_norm = entropy / (self.entropy_ema + 1e-8)
        value_var_norm = value_var / (self.value_var_ema + 1e-8)
        model_uncertainty_norm = model_uncertainty / (self.model_error_ema + 1e-8)
        
        # Weighted combination of uncertainty sources
        uncertainty = 0.5 * entropy_norm + 0.3 * value_var_norm + 0.2 * model_uncertainty_norm
        uncertainty = torch.nan_to_num(uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.clamp(uncertainty, 0.0, 2.0)
    
    def compute_beta(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Compute β gating parameter based on uncertainty."""
        # β increases with uncertainty, capped at beta_max
        beta = self.beta_max * torch.sigmoid(2.0 * (uncertainty - self.uncertainty_threshold))
        return torch.clamp(beta, 0.0, self.beta_max)
    
    def update_kl_constraint(self, kl_divergence: torch.Tensor) -> None:
        """Update Lagrange multiplier for target KL constraint."""
        if kl_divergence.numel() > 1:
            kl_mean = kl_divergence.mean()
        else:
            kl_mean = kl_divergence
        # Use detached mean and update without building a graph
        with torch.no_grad():
            kl_mean_det = kl_mean.detach()
            kl_error = kl_mean_det - self.delta_target
            self.lambda_kl = torch.clamp(
                self.lambda_kl + self.kl_lr * kl_error,
                0.0, 10.0
            )
        
        # Track history for monitoring
        self.kl_history.append(float(kl_mean.detach()))
        if len(self.kl_history) > 1000:
            self.kl_history.pop(0)
    
    def get_kl_penalty_coeff(self) -> float:
        """Get current KL penalty coefficient."""
        return float(self.lambda_kl.detach())
    
    def get_metrics(self) -> Dict[str, float]:
        """Get gating metrics for monitoring."""
        avg_kl = sum(self.kl_history[-100:]) / max(len(self.kl_history[-100:]), 1)
        avg_beta = sum(self.beta_history[-100:]) / max(len(self.beta_history[-100:]), 1) if self.beta_history else 0.0
        
        return {
            'uncertainty_gate/avg_kl': avg_kl,
            'uncertainty_gate/target_kl': self.delta_target,
            'uncertainty_gate/lambda_kl': float(self.lambda_kl.detach()),
            'uncertainty_gate/avg_beta': avg_beta,
            'uncertainty_gate/entropy_ema': float(self.entropy_ema.detach()),
            'uncertainty_gate/value_var_ema': float(self.value_var_ema.detach()),
        }


