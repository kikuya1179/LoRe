from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DreamerV3ActionSpec:
    n: int


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        # Use LazyLinear to register parameters before optimizer creation
        # (materialized on first forward with correct in_features)
        self.fc = nn.LazyLinear(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
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
    def __init__(self, obs_channels: int = 1, latent_dim: int = 256, n_actions: int = 7) -> None:
        super().__init__()
        self.encoder = ConvEncoder(obs_channels, latent_dim)
        # Posterior encoder-driven RSSM (teacher states)
        self.rssm = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.decoder = ConvDecoder(obs_channels, latent_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )
        # Action-conditioned transition for imagination (learned to mimic posterior)
        self.n_actions = int(n_actions)
        self.trans_cell = nn.GRUCell(input_size=latent_dim + self.n_actions, hidden_size=latent_dim)

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs_seq: [B, T, C, H, W]
        B, T = obs_seq.size(0), obs_seq.size(1)
        
        # Fast path for T=1 (single timestep)
        if T == 1:
            # Direct computation without loops
            enc = self.encoder(obs_seq[:, 0])  # [B, latent]
            h0 = torch.zeros(1, B, enc.size(-1), device=enc.device)
            h_seq, _ = self.rssm(enc.unsqueeze(1), h0)  # [B,1,latent]
            h = h_seq.squeeze(1)  # [B, latent]
            recon = self.decoder(h).unsqueeze(1)  # [B,1,C,H,W]
            reward_pred = self.reward_head(h).squeeze(-1)  # [B]
            return h_seq, recon, reward_pred.unsqueeze(1)
        
        # General path for T > 1
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

    def transition(self, h: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """One-step action-conditioned transition in latent space.
        h: [B, latent], action_idx: [B] (long)
        """
        B = h.size(0)
        one_hot = F.one_hot(action_idx.long(), num_classes=self.n_actions).float()
        inp = torch.cat([h, one_hot.to(h.dtype)], dim=-1)
        h_next = self.trans_cell(inp, h)
        return h_next

    def dynamics_loss(self, h_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training for the transition to match posterior RSSM states.
        h_seq: [B,T,latent], action_seq: [B,T]
        """
        B, T, D = h_seq.shape
        if T < 2:
            return torch.tensor(0.0, device=h_seq.device)
        h_pred = h_seq[:, 0]  # start from first posterior state
        loss = 0.0
        steps = 0
        for t in range(T - 1):
            a_t = action_seq[:, t]
            h_pred = self.transition(h_pred, a_t)
            target = h_seq[:, t + 1].detach()
            loss = loss + F.mse_loss(h_pred, target)
            steps += 1
        return loss / max(steps, 1)


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
        # Align dtype to actor weights under autocast
        try:
            target_dtype = next(self.actor.parameters()).dtype
            if actor_input.dtype != target_dtype:
                actor_input = actor_input.to(target_dtype)
        except StopIteration:
            pass
        
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
        # Align dtype to critic weights under autocast
        try:
            target_dtype = next(self.critic.parameters()).dtype
            if h.dtype != target_dtype:
                h = h.to(target_dtype)
        except StopIteration:
            pass
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

        self.world = WorldModel(obs_channels=obs_channels, latent_dim=latent_dim, n_actions=self.n_actions)
        self.ac = ActorCritic(latent_dim=latent_dim, n_actions=self.n_actions, 
                             llm_features_dim=llm_features_dim)
        
        # Set agent reference for uncertainty gate access
        self.ac._agent_ref = self
        
        # Enhanced LLM integration with uncertainty gating (LoRe specification)
        from ..conf import LoReConfig
        lore_cfg = getattr(model_cfg, 'lore', LoReConfig()) if hasattr(model_cfg, 'lore') else LoReConfig()
        
        # Store LoRe configuration 
        self.lore_cfg = lore_cfg
        self.mix_in_imagination = getattr(lore_cfg, "mix_in_imagination", False)
        
        self.uncertainty_gate = UncertaintyGate(
            beta_max=getattr(lore_cfg, "beta_max", 0.3),
            delta_target=getattr(lore_cfg, "delta_target", 0.1),
            kl_lr=getattr(lore_cfg, "kl_lr", 1e-3),
            uncertainty_threshold=getattr(lore_cfg, "uncertainty_threshold", 0.5),
            beta_warmup_steps=getattr(lore_cfg, "beta_warmup_steps", 5000),
            hysteresis_tau_low=getattr(lore_cfg, "hysteresis_tau_low", 0.4),
            hysteresis_tau_high=getattr(lore_cfg, "hysteresis_tau_high", 0.6),
            beta_dropout_p=getattr(lore_cfg, "beta_dropout_p", 0.05)
        )
        # Actor warmup params from config
        self.actor_warmup_steps = int(getattr(model_cfg, 'actor_warmup_steps', getattr(getattr(self, 'lore_cfg', object()), 'actor_warmup_steps', 2000)))
        self.actor_anneal_steps = int(getattr(model_cfg, 'actor_anneal_steps', getattr(getattr(self, 'lore_cfg', object()), 'actor_anneal_steps', 2000)))
        
        # PriorNet reference for imagination (set externally)
        self.priornet_distiller = None
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

        # Include all components in parameters and ensure GPU placement
        self.params = list(self.world.parameters()) + list(self.ac.parameters())
        if self.critic_ensemble is not None:
            self.params += list(self.critic_ensemble.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self._enc_cache = ConvEncoder(in_channels=obs_channels, latent_dim=latent_dim)  # for single-step act
        
        # Force all components to device
        self.to(device)
        # Store device reference for later use
        self.device = device
        
        # Verify GPU placement safely
        if device.type == "cuda":
            gpu_components = []
            try:
                if next(self.world.parameters()).is_cuda:
                    gpu_components.append("WorldModel")
            except StopIteration:
                pass
            try:
                if next(self.ac.parameters()).is_cuda:
                    gpu_components.append("ActorCritic")
            except StopIteration:
                pass
            try:
                if next(self._enc_cache.parameters()).is_cuda:
                    gpu_components.append("Encoder")
            except StopIteration:
                pass
            try:
                if self.critic_ensemble and next(self.critic_ensemble.parameters()).is_cuda:
                    gpu_components.append("CriticEnsemble")
            except StopIteration:
                pass
            print(f"[GPU] DreamerV3Agent components on GPU: {', '.join(gpu_components)}")

        # AMP / GradScaler for CUDA acceleration
        self._amp_enabled = device.type == "cuda"
        try:
            self._scaler = torch.amp.GradScaler('cuda', enabled=self._amp_enabled)
        except Exception:
            self._amp_enabled = False
            self._scaler = None  # type: ignore

    def to(self, device: torch.device) -> None:
        """Move all components to device with verification."""
        # Move individual components (DreamerV3Agent doesn't inherit from nn.Module)
        self.world.to(device)
        self.ac.to(device)
        self._enc_cache.to(device)
        self.uncertainty_gate.to(device)
        if self.critic_ensemble is not None:
            self.critic_ensemble.to(device)
        
        # Force move optimizer state if exists
        if hasattr(self, 'opt') and len(self.opt.state) > 0:
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        
        # Update stored device reference
        self.device = device

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
    def act(self, obs: torch.Tensor, precomputed_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        # obs: [B,C,H,W]
        # Do not toggle train/eval to avoid races with learner thread
        # Inference-only path
        if precomputed_h is not None:
            # Use precomputed hidden state to avoid double encoding/RSSM
            h = precomputed_h
        else:
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
        obs = batch.get("observation").to(self.device, non_blocking=True)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        actions = batch.get("action").to(self.device, non_blocking=True)
        # 形状 [B] に正規化し、ロング型へ
        if actions.dtype != torch.long:
            actions = actions.to(torch.long)
        while actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        # Robust reshape: allow [B], [B,1], or [B,1,1] -> [B]
        rewards = batch.get("reward").to(self.device, non_blocking=True).float().view(-1)
        dones = None
        if "done" in batch.keys(True, True):
            d = batch.get("done").to(self.device, non_blocking=True).float()
            dones = d.view(-1)

        try:
            autocast_ctx = torch.amp.autocast('cuda', enabled=self._amp_enabled)
        except Exception:
            # Fallback for older versions
            from torch.cuda.amp import autocast as _old_autocast
            autocast_ctx = _old_autocast(enabled=self._amp_enabled)

        t_fwd = time.perf_counter()
        with autocast_ctx:
            # Use a pseudo-sequence of length 1 for world model consistency
            obs_seq = obs.unsqueeze(1)  # [B,1,C,H,W]
            h_seq, recon_seq, reward_pred_seq = self.world(obs_seq)
            h = h_seq[:, 0]
            recon = recon_seq[:, 0]
            reward_pred = reward_pred_seq[:, 0]

            # Reconstruction loss (likelihood surrogate)
            if recon.shape[-2:] != obs.shape[-2:]:
                # Resize obs to decoder output spatial size for a valid comparison
                obs_for_loss = F.interpolate(obs, size=recon.shape[-2:], mode="area")
            else:
                obs_for_loss = obs
            recon_loss = F.mse_loss(recon, obs_for_loss)
            # Reward prediction loss
            reward_loss = F.mse_loss(reward_pred, rewards)

        # Extract LLM data and compute enhanced integration
        llm_features = None
        llm_logits = None
        llm_mask = None
        uncertainty = None
        beta_values = None
        
        if "llm_features" in batch.keys(True, True) and self.use_llm_features:
            llm_features = batch.get("llm_features").to(self.device, non_blocking=True)
            if llm_features.numel() > 0:  # Check if not empty
                # Normalize features to [-1, 1] range
                llm_features = torch.clamp(llm_features, -3.0, 3.0) / 3.0
        
        if "llm_prior_logits" in batch.keys(True, True):
            llm_logits = batch.get("llm_prior_logits").to(self.device, non_blocking=True)
        
        if "llm_mask" in batch.keys(True, True):
            # Convert from RBExtra bitmask format if needed
            mask_bits = batch.get("llm_mask").to(self.device, non_blocking=True)
            if mask_bits.dim() == 1:  # Bitmask format
                llm_mask = self._bitmask_to_tensor(mask_bits, self.n_actions)
        
        # Compute values V(s_t)
        if self.critic_ensemble is not None:
            values, ensemble_values = self.ac.value(h, return_ensemble=True)
            value_variance = torch.var(ensemble_values, dim=0)  # [batch_size]
        else:
            values = self.ac.value(h)
            value_variance = None
        values = values.unsqueeze(-1)  # [B,1]

        # Bootstrap target with V(s_{t+1}) from next observation if available
        values_next = values  # fallback
        try:
            if ("next", "observation") in batch.keys(True, True):
                next_obs = batch.get(("next", "observation")).to(self.device, non_blocking=True)
                if next_obs.dim() == 3:
                    next_obs = next_obs.unsqueeze(0)
                # Encode next obs and get next latent/state
                with torch.no_grad():
                    z_next = self.world.encoder(next_obs)
                    h_next_seq, _ = self.world.rssm(
                        z_next.unsqueeze(1), torch.zeros(1, z_next.size(0), z_next.size(-1), device=z_next.device)
                    )
                    h_next = h_next_seq[:, 0]
                    v_next = self.ac.value(h_next).unsqueeze(-1)  # [B,1]
                    values_next = v_next
        except Exception:
            pass
        
        # Disable LLM mixing during learning unless explicitly enabled for imagination
        if not getattr(self, 'lore_cfg', None) or not getattr(self.lore_cfg, 'mix_in_imagination', False):
            llm_logits_for_update = None
            llm_mask_for_update = None
        else:
            llm_logits_for_update = llm_logits
            llm_mask_for_update = llm_mask

        # Policy logits with uncertainty-based LLM integration (controlled by config)
        logits_result = self.ac.policy_logits(
            h, values, llm_features, llm_logits_for_update, llm_mask_for_update,
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
            target_v = self._td0_target(rewards, values_next.detach(), dones).squeeze(-1)
        td_error = (target_v - values.squeeze(-1))
        # Advantage stabilization: avoid normalization collapse for tiny batches
        advantage = td_error.detach()
        try:
            if advantage.numel() >= 4:
                adv_mean = advantage.mean()
                adv_std = advantage.std(unbiased=False).clamp_min(1e-6)
                advantage = (advantage - adv_mean) / adv_std
            # Always clip to limit spikes
            advantage = advantage.clamp(-5.0, 5.0)
        except Exception:
            advantage = advantage.clamp(-5.0, 5.0)

        # Robust targets shape
        target_v = target_v.view_as(values.squeeze(-1))
        policy_loss = -(logp_act * advantage).mean()
        value_loss = F.mse_loss(values.squeeze(-1), target_v)

        # Enhanced KL control following LoRe specification
        kl_divergence = torch.tensor(0.0, device=obs.device)
        kl_penalty = torch.tensor(0.0, device=obs.device)
        
        if llm_logits_for_update is not None:
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
        # 強めの探索: entropy_coef はランタイムでトレーナーから更新される
        # Importance sampling weights from PER (if provided)
        per_weights = None
        try:
            if "_sample_weights" in batch.keys(True, True):
                per_weights = batch.get("_sample_weights").to(obs.device)
                if per_weights.dim() == 2 and per_weights.size(1) == 1:
                    per_weights = per_weights.squeeze(1)
        except Exception:
            per_weights = None

        if per_weights is not None:
            # 重み付きの損失（値・方策）。モデル再構成/報酬は均等重みのまま。
            policy_loss = -(logp_act * advantage * per_weights).mean()
            value_loss = F.mse_loss(values.squeeze(-1), target_v, reduction='none')
            value_loss = (value_loss * per_weights).mean()
        
        loss_ac = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy + kl_penalty
        loss = loss_model + loss_ac

        fwd_ms = (time.perf_counter() - t_fwd) * 1000.0

        self.opt.zero_grad(set_to_none=True)
        t_bwd = time.perf_counter()
        grad_global_norm = torch.tensor(0.0, device=obs.device)
        if self._amp_enabled and getattr(self, "_scaler", None) is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.opt)
            grad_global_norm = torch.nn.utils.clip_grad_norm_(self.params, 2.0)
            self._scaler.step(self.opt)
            self._scaler.update()
        else:
            loss.backward()
            grad_global_norm = torch.nn.utils.clip_grad_norm_(self.params, 2.0)
            self.opt.step()
        bwd_ms = (time.perf_counter() - t_bwd) * 1000.0

        # Collect enhanced metrics
        # LR (first param group)
        try:
            lr0 = float(self.opt.param_groups[0].get('lr', 0.0))
        except Exception:
            lr0 = 0.0

        # Basic stats for monitoring learning signal
        adv_mean = float(advantage.mean().detach().cpu())
        adv_std = float(advantage.std(unbiased=False).detach().cpu())
        val_mean = float(values.mean().detach().cpu())
        val_std = float(values.std(unbiased=False).detach().cpu())
        rew_mean = float(rewards.mean().detach().cpu())
        rew_std = float(rewards.std(unbiased=False).detach().cpu())

        # TD error stats and explained variance of value
        td_abs_mean = float(td_error.abs().mean().detach().cpu())
        td_std = float(td_error.std(unbiased=False).detach().cpu())
        try:
            var_y = float(torch.var(target_v.detach(), correction=0).cpu())
            var_res = float(torch.var((target_v - values.squeeze(-1)).detach(), correction=0).cpu())
            value_ev = 1.0 - (var_res / (var_y + 1e-8))
        except Exception:
            value_ev = 0.0

        # PSNR for reconstruction (assumes 0-1 range)
        try:
            mse = float(recon_loss.detach().cpu())
            world_psnr = 10.0 * float(torch.log10(torch.tensor(1.0 / max(mse, 1e-8))))
        except Exception:
            world_psnr = 0.0

        metrics = {
            "loss/model_recon": float(recon_loss.detach().cpu()),
            "loss/model_reward": float(reward_loss.detach().cpu()),
            "loss/policy": float(policy_loss.detach().cpu()),
            "loss/value": float(value_loss.detach().cpu()),
            # expose policy entropy under policy/entropy tag for dashboard spec
            "policy/entropy": float(entropy.detach().cpu()),
            "loss/kl_divergence": float(kl_divergence.mean().detach().cpu()) if kl_divergence.numel() > 1 else float(kl_divergence.detach().cpu()),
            # grads & lr
            "optim/grad_global_norm": float(grad_global_norm.detach().cpu()),
            "optim/lr": lr0,
            # stats
            "advantage/mean": adv_mean,
            "advantage/std": adv_std,
            "value/mean": val_mean,
            "value/std": val_std,
            "reward/batch_mean": rew_mean,
            "reward/batch_std": rew_std,
            "value/td_abs_mean": td_abs_mean,
            "value/td_std": td_std,
            "value/explained_variance": float(value_ev),
            "world/psnr_db": world_psnr,
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
            try:
                if uncertainty.numel() > 1:
                    unc_std = torch.std(uncertainty, unbiased=False)
                else:
                    unc_std = torch.tensor(0.0, device=uncertainty.device)
                metrics["uncertainty/std"] = float(unc_std.detach().cpu())
            except Exception:
                metrics["uncertainty/std"] = 0.0
        if beta_values is not None:
            metrics["beta/mean"] = float(beta_values.mean().detach().cpu())
            metrics["beta/std"] = float(beta_values.std().detach().cpu())
        
        # Add uncertainty gate internal metrics
        gate_metrics = self.uncertainty_gate.get_metrics()
        metrics.update(gate_metrics)
        
        return metrics

    def update_sequence(self, batch: Dict[str, torch.Tensor]) -> dict:
        """Sequence training: obs_seq [B,T,C,H,W], action_seq [B,T], reward_seq [B,T], done_seq [B,T]."""
        self.world.train()
        self.ac.train()

        obs_seq = batch["observation_seq"].to(self.device, non_blocking=True)  # [B,T,C,H,W]
        next_obs_seq = batch.get("next_observation_seq").to(self.device, non_blocking=True) if "next_observation_seq" in batch else None
        actions = batch["action_seq"].to(self.device, non_blocking=True)        # [B,T]
        rewards = batch["reward_seq"].to(self.device, non_blocking=True)        # [B,T]
        dones = batch["done_seq"].to(self.device, non_blocking=True)            # [B,T]

        B, T = obs_seq.size(0), obs_seq.size(1)

        try:
            autocast_ctx = torch.amp.autocast('cuda', enabled=self._amp_enabled)
        except Exception:
            from torch.cuda.amp import autocast as _old_autocast
            autocast_ctx = _old_autocast(enabled=self._amp_enabled)

        with autocast_ctx:
            h_seq, recon_seq, reward_pred_seq = self.world(obs_seq)
            # reconstruction loss over all steps
            recon_loss = F.mse_loss(recon_seq, obs_seq)
            reward_loss = F.mse_loss(reward_pred_seq.squeeze(-1), rewards)
            # train dynamics to imitate posterior transitions (Dreamer風の教師あり)
            dyn_loss = self.world.dynamics_loss(h_seq.detach(), actions)

        # Values for each step
        h_flat = h_seq.reshape(B * T, -1)
        values = self.ac.value(h_flat).view(B, T)

        # Bootstrap value for last next observation
        if next_obs_seq is not None:
            with torch.no_grad():
                h_next_last, _, _ = self.world(next_obs_seq[:, -1:].contiguous())  # [B,1,D]
                v_last = self.ac.value(h_next_last[:, 0]).detach()  # [B]
        else:
            v_last = torch.zeros(B, device=self.device)

        # GAE(λ) advantages and returns
        lam = 0.95
        gamma = self.gamma
        advantages = torch.zeros(B, T, device=self.device)
        last_gae = torch.zeros(B, device=self.device)
        for t in reversed(range(T)):
            v_t = values[:, t]
            v_tp1 = values[:, t + 1] if t + 1 < T else v_last
            r_t = rewards[:, t]
            d_t = dones[:, t]
            delta = r_t + gamma * (1.0 - d_t) * v_tp1 - v_t
            last_gae = delta + gamma * lam * (1.0 - d_t) * last_gae
            advantages[:, t] = last_gae
        returns = advantages + values

        # Policy logits per step
        logits = self.ac.policy_logits(h_flat).view(B, T, -1)
        logp = F.log_softmax(logits, dim=-1)
        logp_act = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        adv = advantages.detach()
        # Normalize per batch-time if十分大きい
        if adv.numel() >= 128:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False).clamp_min(1e-6))
        adv = adv.clamp(-5.0, 5.0)

        # Actor warmup/anneal to avoid early collapse
        global_step = float(self.uncertainty_gate.current_step)
        warm = getattr(self, 'actor_warmup_steps', 2000)
        anneal = getattr(self, 'actor_anneal_steps', 2000)
        if global_step < warm:
            actor_scale = global_step / max(1.0, warm)
        elif global_step < warm + anneal:
            # cosine anneal to 1.0
            import math
            ratio = (global_step - warm) / max(1.0, anneal)
            actor_scale = 0.5 * (1.0 - math.cos(math.pi * ratio))
        else:
            actor_scale = 1.0
        policy_loss = actor_scale * (-(logp_act * adv).mean())
        value_loss = F.mse_loss(values, returns)
        entropy = -(torch.exp(logp) * logp).sum(-1).mean()

        # World model diagnostics
        try:
            mse_val = float(recon_loss.detach().cpu())
            world_psnr = 10.0 * float(torch.log10(torch.tensor(1.0 / max(mse_val, 1e-8))))
        except Exception:
            world_psnr = 0.0
        reward_mae = float(torch.mean(torch.abs(reward_pred_seq.squeeze(-1) - rewards)).detach().cpu())
        # Explained variance for value and reward
        try:
            var_y_val = float(torch.var(returns.detach().reshape(-1), correction=0).cpu())
            var_res_val = float(torch.var((returns - values).detach().reshape(-1), correction=0).cpu())
            value_ev = 1.0 - (var_res_val / (var_y_val + 1e-8))
        except Exception:
            value_ev = 0.0
        try:
            var_y_r = float(torch.var(rewards.detach().reshape(-1), correction=0).cpu())
            var_res_r = float(torch.var((rewards - reward_pred_seq.squeeze(-1)).detach().reshape(-1), correction=0).cpu())
            reward_ev = 1.0 - (var_res_r / (var_y_r + 1e-8))
        except Exception:
            reward_ev = 0.0

        loss_model = recon_loss + reward_loss + 0.5 * dyn_loss
        loss_ac = policy_loss + 1.0 * value_loss - self.entropy_coef * entropy
        loss = loss_model + loss_ac

        self.opt.zero_grad(set_to_none=True)
        if self._amp_enabled and getattr(self, "_scaler", None) is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.opt)
            grad_global_norm = torch.nn.utils.clip_grad_norm_(self.params, 2.0)
            self._scaler.step(self.opt)
            self._scaler.update()
        else:
            loss.backward()
            grad_global_norm = torch.nn.utils.clip_grad_norm_(self.params, 2.0)
            self.opt.step()

        # Optimizer LR
        try:
            lr0 = float(self.opt.param_groups[0].get('lr', 0.0))
        except Exception:
            lr0 = 0.0

        return {
            "loss/model_recon": float(recon_loss.detach().cpu()),
            "loss/model_reward": float(reward_loss.detach().cpu()),
            "loss/model_dyn": float(dyn_loss.detach().cpu()),
            "loss/policy": float(policy_loss.detach().cpu()),
            "loss/value": float(value_loss.detach().cpu()),
            "policy/entropy": float(entropy.detach().cpu()),
            "value/mean": float(values.mean().detach().cpu()),
            "value/explained_variance": float(value_ev),
            "reward/mae": float(reward_mae),
            "reward/explained_variance": float(reward_ev),
            "world/psnr_db": float(world_psnr),
            "world/recon_mse": float(mse_val),
            "optim/grad_global_norm": float(grad_global_norm.detach().cpu()) if 'grad_global_norm' in locals() else 0.0,
            "optim/lr": lr0,
        }
    
    def _generate_imagination_priors(self, h_seq: torch.Tensor, context_batch: List[Dict] = None) -> Optional[torch.Tensor]:
        """Generate PriorNet logits for imagination rollouts."""
        if not self.mix_in_imagination or not self.priornet_distiller:
            return None
        
        # Use PriorNet to generate priors for latent states
        if not self.priornet_distiller.should_use_priornet():
            return None
        
        batch_size, seq_len = h_seq.shape[:2]
        prior_logits_list = []
        
        # For each step in the imagination sequence
        for t in range(seq_len):
            h_t = h_seq[:, t]  # [batch_size, latent_dim]
            
            # Create dummy observations from latent states (simplified)
            # In practice, you'd decode h_t back to observation space
            dummy_obs = torch.zeros(batch_size, 1, 64, 64, device=h_t.device)
            
            # Create dummy context
            dummy_context = {'mission': 'explore', 'remaining_steps': 50}
            
            try:
                # Get PriorNet prediction for this latent state  
                with torch.no_grad():
                    # Convert to numpy for PriorNet interface
                    obs_np = dummy_obs[0].cpu().numpy()  # Take first in batch
                    prior_response = self.priornet_distiller.predict(obs_np, dummy_context)
                    
                    if prior_response and 'policy' in prior_response:
                        prior_logits = torch.tensor(
                            prior_response['policy']['logits'], 
                            device=h_t.device, 
                            dtype=torch.float32
                        ).unsqueeze(0).expand(batch_size, -1)
                        prior_logits_list.append(prior_logits)
                    else:
                        # Fallback to zero logits if PriorNet fails
                        zero_logits = torch.zeros(batch_size, self.n_actions, device=h_t.device)
                        prior_logits_list.append(zero_logits)
                        
            except Exception:
                # Fallback on any error
                zero_logits = torch.zeros(batch_size, self.n_actions, device=h_t.device)
                prior_logits_list.append(zero_logits)
        
        if prior_logits_list:
            return torch.stack(prior_logits_list, dim=1)  # [batch_size, seq_len, n_actions]
        
        return None
    
    def set_priornet_distiller(self, distiller):
        """Set PriorNet distiller reference for imagination integration."""
        self.priornet_distiller = distiller
    
    def _bitmask_to_tensor(self, mask_bits: torch.Tensor, n_actions: int) -> torch.Tensor:
        """Convert bitmask to action mask tensor."""
        batch_size = mask_bits.size(0)
        mask = torch.zeros(batch_size, n_actions, device=mask_bits.device)
        
        for i in range(min(n_actions, 64)):  # Support up to 64 actions
            bit_mask = (mask_bits & (1 << i)) != 0
            mask[:, i] = bit_mask.float()
        
        return mask


class UncertaintyGate:
    """Enhanced uncertainty-based β gating for LLM integration following LoRe specification."""
    
    def __init__(self, beta_max: float = 0.3, delta_target: float = 0.1, 
                 kl_lr: float = 1e-3, uncertainty_threshold: float = 0.5,
                 beta_warmup_steps: int = 5000, hysteresis_tau_low: float = 0.4,
                 hysteresis_tau_high: float = 0.6, beta_dropout_p: float = 0.05):
        self.beta_max = beta_max
        self.delta_target = delta_target
        self.kl_lr = kl_lr
        self.uncertainty_threshold = uncertainty_threshold
        
        # Enhanced parameters from LoRe specification
        self.beta_warmup_steps = beta_warmup_steps
        self.hysteresis_tau_low = hysteresis_tau_low
        self.hysteresis_tau_high = hysteresis_tau_high
        self.beta_dropout_p = beta_dropout_p
        
        # State tracking for enhanced features
        self.current_step = 0
        self.uncertainty_state = 'low'  # 'low' or 'high'
        
        # EMA for uncertainty threshold computation
        self.uncertainty_ema = torch.tensor(0.5, dtype=torch.float32)
        self.uncertainty_queue = []  # For percentile-based threshold
        self.queue_size = 5000
        
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
        self.warmup_schedule = []
    
    def to(self, device: torch.device) -> None:
        """Move tensors to device."""
        self.lambda_kl = self.lambda_kl.to(device)
        self.entropy_ema = self.entropy_ema.to(device)
        self.value_var_ema = self.value_var_ema.to(device)
        self.model_error_ema = self.model_error_ema.to(device)
        
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
        """Enhanced β computation with warmup, hysteresis, and dropout."""
        batch_size = uncertainty.size(0)
        device = uncertainty.device
        
        # Update internal step counter
        self.current_step += 1
        
        # 1. Beta warmup schedule
        if self.current_step < self.beta_warmup_steps:
            warmup_factor = self.current_step / self.beta_warmup_steps
            self.warmup_schedule.append(warmup_factor)
            if len(self.warmup_schedule) > 1000:
                self.warmup_schedule.pop(0)
        else:
            warmup_factor = 1.0
        
        # 2. Update uncertainty queue and compute adaptive threshold
        with torch.no_grad():
            unc_mean = uncertainty.mean().item()
            self.uncertainty_queue.append(unc_mean)
            if len(self.uncertainty_queue) > self.queue_size:
                self.uncertainty_queue.pop(0)
            
            # Update EMA
            self.uncertainty_ema = (self.ema_decay * self.uncertainty_ema + 
                                  (1 - self.ema_decay) * unc_mean)
            
            # Compute adaptive threshold (median of recent samples)
            if len(self.uncertainty_queue) >= 100:
                import statistics
                adaptive_threshold = statistics.median(self.uncertainty_queue[-1000:])
            else:
                adaptive_threshold = self.uncertainty_threshold
        
        # 3. Hysteresis-based state switching
        with torch.no_grad():
            if self.uncertainty_state == 'low' and unc_mean > self.hysteresis_tau_high:
                self.uncertainty_state = 'high'
            elif self.uncertainty_state == 'high' and unc_mean < self.hysteresis_tau_low:
                self.uncertainty_state = 'low'
        
        # 4. Compute base β with adaptive threshold
        if self.uncertainty_state == 'high':
            base_beta = self.beta_max * torch.sigmoid(2.0 * (uncertainty - adaptive_threshold))
        else:
            # Lower β when in low uncertainty state
            base_beta = self.beta_max * 0.5 * torch.sigmoid(2.0 * (uncertainty - adaptive_threshold))
        
        # 5. Apply warmup factor
        beta = base_beta * warmup_factor
        
        # 6. Beta dropout for training stability
        if self.training and torch.rand(1).item() < self.beta_dropout_p:
            beta = torch.zeros_like(beta)
        
        beta = torch.clamp(beta, 0.0, self.beta_max)
        
        # Store for metrics
        self.beta_history.extend(beta.detach().cpu().tolist())
        if len(self.beta_history) > 1000:
            self.beta_history = self.beta_history[-500:]
        
        return beta
    
    @property
    def training(self) -> bool:
        """Check if in training mode (simplified heuristic)."""
        return True  # Could be set by parent agent
    
    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self._training = training
    
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
    
    def get_metrics(self) -> dict[str, float]:
        """Get enhanced gating metrics for monitoring."""
        avg_kl = sum(self.kl_history[-100:]) / max(len(self.kl_history[-100:]), 1)
        avg_beta = sum(self.beta_history[-100:]) / max(len(self.beta_history[-100:]), 1) if self.beta_history else 0.0
        beta_std = 0.0
        if len(self.beta_history) >= 10:
            import statistics
            beta_std = statistics.stdev(self.beta_history[-100:])
        
        # Compute current warmup factor
        current_warmup = min(1.0, self.current_step / self.beta_warmup_steps) if self.beta_warmup_steps > 0 else 1.0
        
        # Adaptive threshold
        adaptive_threshold = self.uncertainty_threshold
        if len(self.uncertainty_queue) >= 100:
            import statistics
            adaptive_threshold = statistics.median(self.uncertainty_queue[-1000:])
        
        return {
            'uncertainty_gate/avg_kl': avg_kl,
            'uncertainty_gate/target_kl': self.delta_target,
            'uncertainty_gate/lambda_kl': float(self.lambda_kl.detach()),
            'uncertainty_gate/avg_beta': avg_beta,
            'uncertainty_gate/beta_std': beta_std,
            'uncertainty_gate/entropy_ema': float(self.entropy_ema.detach()),
            'uncertainty_gate/value_var_ema': float(self.value_var_ema.detach()),
            'uncertainty_gate/uncertainty_ema': float(self.uncertainty_ema.detach()),
            'uncertainty_gate/adaptive_threshold': adaptive_threshold,
            'uncertainty_gate/warmup_progress': current_warmup,
            'uncertainty_gate/uncertainty_state': 1.0 if self.uncertainty_state == 'high' else 0.0,
            'uncertainty_gate/current_step': float(self.current_step),
        }


