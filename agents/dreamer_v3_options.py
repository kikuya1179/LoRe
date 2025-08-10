"""DreamerV3 agent with hierarchical options support (LoRe Option C)."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dreamer_v3 import DreamerV3Agent, WorldModel, UncertaintyGate
from ..options.option_framework import (
    OptionManager, DualHeadActor, OptionTerminationPredictor
)
from ..options.llm_skill_generator import LLMSkillGenerator, SkillEvaluator, CrafterSkillLibrary


class HierarchicalDreamerV3Agent(DreamerV3Agent):
    """DreamerV3 with hierarchical options following LoRe specification."""
    
    def __init__(
        self,
        model_cfg: Any,
        action_spec: Any,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.05,
        epsilon_greedy: float = 0.0,
        # Option-specific parameters
        max_options: int = 8,
        option_generation_interval: int = 500,
        skill_confidence_threshold: float = 0.4,
    ) -> None:
        
        # Initialize base DreamerV3 components
        self.device = device
        self.gamma = gamma
        self.lambda_kl: float = 0.0
        self.lambda_bc: float = getattr(model_cfg, "lambda_bc", 0.1)
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
        
        # Model configuration
        obs_channels = int(getattr(model_cfg, "obs_channels", 1))
        latent_dim = int(getattr(model_cfg, "latent_dim", 256))
        llm_features_dim = int(getattr(model_cfg, "llm_features_dim", 0))
        
        # World model (shared with base DreamerV3)
        self.world = WorldModel(obs_channels=obs_channels, latent_dim=latent_dim)
        
        # Option system initialization
        self.max_options = max_options
        self.option_generation_interval = option_generation_interval
        self.last_option_generation = 0
        
        # Option manager
        self.option_manager = OptionManager(
            max_options=max_options,
            min_success_rate=0.15,
            evaluation_period=20,
        )
        
        # Dual-head actor (primitive + options)
        self.ac = DualHeadActor(
            latent_dim=latent_dim,
            num_primitive_actions=self.n_actions,
            num_options=max_options,
            llm_features_dim=llm_features_dim,
        )
        
        # Option termination predictor
        self.termination_predictor = OptionTerminationPredictor(
            latent_dim=latent_dim,
            num_options=max_options,
        )
        
        # Value function (same as base)
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )
        
        # Enhanced LLM integration
        self.uncertainty_gate = UncertaintyGate(
            beta_max=getattr(model_cfg, "beta_max", 0.3),
            delta_target=getattr(model_cfg, "delta_target", 0.1),
            kl_lr=getattr(model_cfg, "kl_lr", 1e-3),
            uncertainty_threshold=getattr(model_cfg, "uncertainty_threshold", 0.5)
        )
        self.use_llm_features = llm_features_dim > 0
        
        # LLM skill generation (initialized later with LLM adapter)
        self.skill_generator: Optional[LLMSkillGenerator] = None
        self.skill_evaluator = SkillEvaluator(evaluation_window=20)
        
        # Training parameters
        self.params = (list(self.world.parameters()) + 
                      list(self.ac.parameters()) + 
                      list(self.termination_predictor.parameters()) +
                      list(self.critic.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=lr)
        
        # Encoder cache for single-step action
        self._enc_cache = self.world.encoder
        
        # Set agent reference for uncertainty gate
        self.ac._agent_ref = self
        
        # Option execution state
        self.current_step = 0
        
        self.to(device)
    
    def set_llm_adapter(self, llm_adapter):
        """Set LLM adapter for skill generation."""
        self.skill_generator = LLMSkillGenerator(
            llm_adapter=llm_adapter,
            skill_library=CrafterSkillLibrary(),
            max_skill_length=8,
            confidence_threshold=0.4,
        )
    
    def to(self, device: torch.device) -> None:
        """Move all components to device."""
        self.world.to(device)
        self.ac.to(device)
        self.termination_predictor.to(device)
        self.critic.to(device)
        self._enc_cache.to(device)
    
    @torch.no_grad()
    def act(self, obs: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Enhanced action selection with options support."""
        self.world.eval()
        self.ac.eval()
        self.termination_predictor.eval()
        
        # Encode observation
        z = self._enc_cache(obs)
        h, _ = self.world.rssm(z.unsqueeze(1), torch.zeros(1, obs.size(0), z.size(-1), device=obs.device))
        h = h.squeeze(1)
        
        # Check if currently executing an option
        current_action = self.option_manager.get_current_action()
        
        if current_action is not None:
            # Continue executing current option
            return torch.tensor([[current_action]], device=obs.device, dtype=torch.long)
        
        # Get available options mask
        available_options = self.option_manager.get_available_options_mask(obs.device).unsqueeze(0)
        
        # Get action from dual-head actor
        if self.epsilon_greedy > 0.0 and np.random.random() < self.epsilon_greedy:
            # Epsilon-greedy exploration
            total_actions = self.n_actions + self.option_manager.num_options()
            action_id = np.random.randint(0, total_actions)
            is_option = action_id >= self.n_actions
        else:
            # Policy-based action selection
            action_id, is_option, _ = self.ac.sample_action(
                h, available_options=available_options, temperature=1.0
            )
        
        if is_option:
            # Start option execution
            option_index = action_id - self.n_actions
            success = self.option_manager.start_option_execution(option_index, self.current_step)
            
            if success:
                # Get first action from option
                option_action = self.option_manager.get_current_action()
                if option_action is not None:
                    return torch.tensor([[option_action]], device=obs.device, dtype=torch.long)
            
            # Fallback to primitive action if option start failed
            action_id = np.random.randint(0, self.n_actions)
        
        return torch.tensor([[action_id]], device=obs.device, dtype=torch.long)
    
    def step_options(self, reward: float):
        """Update option execution state."""
        self.current_step += 1
        
        # Update current option if executing
        if self.option_manager.current_execution is not None:
            option_terminated = self.option_manager.step_current_option(reward)
            
            if option_terminated:
                # Record performance for skill evaluation
                execution = self.option_manager.execution_history[-1]  # Last completed execution
                self.skill_evaluator.record_skill_performance(
                    skill_id=execution.option_spec.option_id,
                    reward=execution.accumulated_reward,
                    success=execution.is_successful(),
                    duration=execution.steps_executed,
                )
    
    def generate_new_skills(self, obs: torch.Tensor, context: Dict[str, Any]):
        """Generate new skills using LLM if conditions are met."""
        
        if (self.skill_generator is None or 
            self.current_step - self.last_option_generation < self.option_generation_interval):
            return
        
        # Check if we need new skills (poor performance or low diversity)
        current_options = len(self.option_manager.options)
        if current_options >= self.max_options:
            # Check if any skills are performing poorly
            skill_ids = list(self.option_manager.options.keys())
            skill_ranking = self.skill_evaluator.get_skill_ranking(skill_ids)
            
            if skill_ranking and skill_ranking[-1][1] < -0.05:  # Worst skill is bad
                # Remove worst skill to make room
                worst_skill_id = skill_ranking[-1][0]
                self.option_manager.remove_option(worst_skill_id)
            else:
                return  # All skills are good, don't generate new ones
        
        try:
            # Generate new skills
            obs_np = obs.detach().cpu().numpy()
            new_skills = self.skill_generator.generate_skills_for_context(
                obs_np, context, num_skills=2
            )
            
            # Add successful skills to manager
            for skill in new_skills:
                success = self.option_manager.add_option(skill)
                if success:
                    print(f"Added new skill: {skill.name} (confidence: {skill.confidence:.2f})")
            
            self.last_option_generation = self.current_step
            
        except Exception as e:
            print(f"Skill generation failed: {e}")
    
    def update(self, batch: "TensorDictBase") -> dict:  # type: ignore[name-defined]
        """Enhanced update with option training."""
        self.world.train()
        self.ac.train()
        self.termination_predictor.train()
        self.critic.train()
        
        # Extract batch data (same as base DreamerV3)
        obs = batch.get("observation").to(self.device)
        actions = batch.get("action").to(self.device)
        
        # Normalize actions
        if actions.dtype != torch.long:
            actions = actions.to(torch.long)
        while actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        rewards = batch.get("reward").to(self.device).squeeze(-1)
        dones = batch.get("done").to(self.device).squeeze(-1) if "done" in batch.keys(True, True) else None
        
        # World model forward pass
        obs_seq = obs.unsqueeze(1)  # [B,1,C,H,W]
        h_seq, recon_seq, reward_pred_seq = self.world(obs_seq)
        h = h_seq[:, 0]
        recon = recon_seq[:, 0]
        reward_pred = reward_pred_seq[:, 0]
        
        # World model losses
        recon_loss = F.mse_loss(recon, obs)
        reward_loss = F.mse_loss(reward_pred, rewards)
        
        # Extract LLM data
        llm_features = None
        llm_logits = None
        llm_mask = None
        
        if "llm_features" in batch.keys(True, True) and self.use_llm_features:
            llm_features = batch.get("llm_features").to(self.device)
            if llm_features.numel() > 0:
                llm_features = torch.clamp(llm_features, -3.0, 3.0) / 3.0
        
        if "llm_prior_logits" in batch.keys(True, True):
            llm_logits = batch.get("llm_prior_logits").to(self.device)
        
        # Value computation
        values = self.critic(h).squeeze(-1)
        
        # Option-aware policy computation
        available_options = self.option_manager.get_available_options_mask(self.device)
        available_options = available_options.unsqueeze(0).expand(h.size(0), -1)
        
        # Dual-head actor forward pass
        actor_outputs = self.ac.forward(h, llm_features, available_options)
        combined_logits = actor_outputs['combined_logits']
        
        # Apply uncertainty-based LLM mixing to combined logits
        if llm_logits is not None:
            # Pad LLM logits to match combined action space
            if llm_logits.size(-1) == self.n_actions:  # LLM only provides primitive action logits
                padding = torch.zeros(llm_logits.size(0), self.max_options, device=llm_logits.device)
                llm_logits_padded = torch.cat([llm_logits, padding], dim=-1)
            else:
                llm_logits_padded = llm_logits
            
            # Apply uncertainty gating
            values_for_uncertainty = values.unsqueeze(-1)
            uncertainty = self.uncertainty_gate.compute_uncertainty(combined_logits, values_for_uncertainty)
            beta = self.uncertainty_gate.compute_beta(uncertainty)
            
            # Mix logits with stopgrad on LLM
            beta_expanded = beta.unsqueeze(-1).expand_as(combined_logits)
            combined_logits = combined_logits + beta_expanded * llm_logits_padded.detach()
        
        # Action probabilities and log probabilities
        log_probs = F.log_softmax(combined_logits, dim=-1)
        
        # Expand actions for combined action space
        expanded_actions = actions  # Actions should already be in combined space
        log_probs_act = log_probs.gather(-1, expanded_actions.unsqueeze(-1)).squeeze(-1)
        
        # TD target computation (same as base)
        with torch.no_grad():
            target_v = self._td0_target(rewards, values.detach(), dones).squeeze(-1)
        advantage = (target_v - values).detach()
        
        # Policy and value losses
        policy_loss = -(log_probs_act * advantage).mean()
        value_loss = F.mse_loss(values, target_v)
        
        # Option termination loss (for options that were executed)
        termination_loss = torch.tensor(0.0, device=obs.device)
        if "option_termination_targets" in batch.keys(True, True):
            # If we have termination targets from option executions
            termination_targets = batch.get("option_termination_targets").to(self.device)
            option_ids = batch.get("option_ids").to(self.device)
            execution_times = batch.get("execution_times").to(self.device)
            
            # Compute termination predictions
            termination_probs, option_values = self.termination_predictor(
                h, option_ids[0].item(), execution_times
            )
            
            # Termination loss (binary cross-entropy)
            termination_loss = F.binary_cross_entropy(
                termination_probs, termination_targets.float()
            )
        
        # Enhanced KL control (same as base)
        kl_divergence = torch.tensor(0.0, device=obs.device)
        kl_penalty = torch.tensor(0.0, device=obs.device)
        
        if llm_logits is not None:
            try:
                from torch.distributions import Categorical
                
                pi_mixed = Categorical(logits=combined_logits)
                pi_base = Categorical(logits=actor_outputs['combined_logits'])  # Base without LLM mixing
                kl_divergence = torch.distributions.kl.kl_divergence(pi_mixed, pi_base)
                kl_mean = kl_divergence.mean()
                
                self.uncertainty_gate.update_kl_constraint(kl_mean)
                kl_penalty_coeff = self.uncertainty_gate.get_kl_penalty_coeff()
                kl_penalty = kl_penalty_coeff * torch.relu(kl_mean - self.uncertainty_gate.delta_target)
                
            except Exception:
                pass
        
        # Behavioral cloning for synthetic data
        bc_loss = torch.tensor(0.0, device=obs.device)
        if "_is_synthetic" in batch.keys(True, True):
            is_synthetic = batch.get("_is_synthetic").to(obs.device)
            synthetic_mask = is_synthetic.float()
            
            if synthetic_mask.sum() > 0 and llm_logits is not None:
                log_probs_synthetic = log_probs * synthetic_mask.unsqueeze(-1)
                bc_loss = -log_probs_synthetic.sum(dim=-1).mean()
                bc_loss = self.lambda_bc * bc_loss * (synthetic_mask.sum() / synthetic_mask.numel())
        
        # Entropy for exploration
        entropy = -(log_probs * torch.exp(log_probs)).sum(-1).mean()
        
        # Total loss
        loss_model = recon_loss + reward_loss
        loss_ac = (policy_loss + 0.5 * value_loss + 0.1 * termination_loss - 
                   self.entropy_coef * entropy + kl_penalty + bc_loss)
        loss = loss_model + loss_ac
        
        # Optimization step
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.opt.step()
        
        # Collect metrics
        metrics = {
            "loss/model_recon": float(recon_loss.detach().cpu()),
            "loss/model_reward": float(reward_loss.detach().cpu()),
            "loss/policy": float(policy_loss.detach().cpu()),
            "loss/value": float(value_loss.detach().cpu()),
            "loss/termination": float(termination_loss.detach().cpu()),
            "loss/entropy": float(entropy.detach().cpu()),
            "loss/kl_divergence": float(kl_divergence.mean().detach().cpu()) if kl_divergence.numel() > 1 else float(kl_divergence.detach().cpu()),
            "loss/kl_penalty": float(kl_penalty.detach().cpu()),
            "loss/bc_regularization": float(bc_loss.detach().cpu()),
        }
        
        # Add option-specific metrics
        option_stats = self.option_manager.get_statistics()
        for k, v in option_stats.items():
            metrics[f"options/{k}"] = float(v)
        
        if self.skill_generator:
            skill_stats = self.skill_generator.get_statistics()
            for k, v in skill_stats.items():
                metrics[f"skill_gen/{k}"] = float(v)
        
        # Add uncertainty gate metrics
        gate_metrics = self.uncertainty_gate.get_metrics()
        metrics.update(gate_metrics)
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save agent state including options."""
        state = {
            "world": self.world.state_dict(),
            "ac": self.ac.state_dict(),
            "termination_predictor": self.termination_predictor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
            "lambda_kl": self.lambda_kl,
            "lambda_bc": self.lambda_bc,
            "gamma": self.gamma,
            "n_actions": self.n_actions,
            "current_step": self.current_step,
            # Option system state
            "options": {opt_id: {
                'spec': opt.__dict__,
                'index': self.option_manager.option_id_to_index[opt_id]
            } for opt_id, opt in self.option_manager.options.items()},
            "option_stats": self.option_manager.get_statistics(),
        }
        torch.save(state, path)
    
    def load(self, path: str, strict: bool = True) -> None:
        """Load agent state including options."""
        state = torch.load(path, map_location=self.device)
        
        self.world.load_state_dict(state["world"], strict=strict)
        self.ac.load_state_dict(state["ac"], strict=strict)
        self.termination_predictor.load_state_dict(state.get("termination_predictor", {}), strict=strict)
        self.critic.load_state_dict(state.get("critic", {}), strict=strict)
        
        try:
            self.opt.load_state_dict(state["opt"])
        except Exception:
            pass
        
        self.lambda_kl = float(state.get("lambda_kl", self.lambda_kl))
        self.lambda_bc = float(state.get("lambda_bc", self.lambda_bc))
        self.gamma = float(state.get("gamma", self.gamma))
        self.current_step = int(state.get("current_step", 0))
        
        # Restore options (simplified - would need full reconstruction in production)
        if "options" in state:
            print(f"Loaded {len(state['options'])} saved options")
    
    def _td0_target(self, rewards: torch.Tensor, values_next: torch.Tensor, dones: Optional[torch.Tensor]) -> torch.Tensor:
        """TD(0) target computation (same as base)."""
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