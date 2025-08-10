"""Enhanced trainer with LoRe synthetic replay augmentation."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import torch

from ..utils.synthetic_replay import EnhancedReplayBuffer
from ..utils.synthetic_generator import SyntheticExperienceGenerator
from ..llm.enhanced_adapter import EnhancedLLMAdapter, LLMAdapterConfigV2
from ..utils.rnd import RND
from .trainer import Trainer  # Import base trainer


class EnhancedTrainer(Trainer):
    """Enhanced trainer with LoRe synthetic experience augmentation."""
    
    def __init__(self, env, agent, logger, cfg, device) -> None:
        # Initialize base components
        self.env = env
        self.agent = agent
        self.logger = logger
        self.cfg = cfg
        self.device = device
        
        # Enhanced replay buffer with synthetic support
        self.replay = EnhancedReplayBuffer(
            capacity=cfg.train.replay_capacity,
            synthetic_ratio_max=getattr(cfg.train, 'synthetic_ratio_max', 0.25),
            bc_regularization_coeff=getattr(cfg.model, 'lambda_bc', 0.1),
            importance_sampling=True,
            synthetic_weight_decay=getattr(cfg.train, 'synthetic_weight_decay', 0.99),
        )
        
        # Enhanced LLM adapter
        self.llm = EnhancedLLMAdapter(
            LLMAdapterConfigV2(
                enabled=getattr(cfg.train, "use_llm", False),
                model=getattr(cfg.train, "llm_model", "gemini-2.5-flash-lite"),
                features_dim=getattr(cfg.model, "llm_features_dim", 0),
                use_cli=getattr(cfg.train, "llm_use_cli", True),
                batch_size=getattr(cfg.train, "llm_batch_size", 8),
                cache_size=getattr(cfg.train, "llm_cache_size", 1000),
                use_dsl=getattr(cfg.train, "llm_use_dsl", True),
                timeout_s=getattr(cfg.train, "llm_timeout_s", 2.5),
                novelty_threshold=getattr(cfg.train, "llm_novelty_threshold", 0.1),
                td_error_threshold=getattr(cfg.train, "llm_td_error_threshold", 0.5),
                plateau_frames=getattr(cfg.train, "llm_plateau_frames", 10000),
            )
        )
        
        # Synthetic experience generator
        if getattr(cfg.train, "use_llm", False) and hasattr(agent, 'world'):
            self.synthetic_generator = SyntheticExperienceGenerator(
                world_model=agent.world,
                llm_adapter=self.llm,
                device=device,
                max_rollout_length=getattr(cfg.train, 'synthetic_rollout_length', 5),
                confidence_threshold=getattr(cfg.train, 'synthetic_confidence_threshold', 0.3),
                success_reward_threshold=getattr(cfg.train, 'synthetic_success_threshold', 0.1),
                synthetic_execution_prob=getattr(cfg.train, 'synthetic_execution_prob', 0.2),
            )
        else:
            self.synthetic_generator = None
        
        # Initialize environment observation spec
        try:
            self.obs_key = "pixels" if "pixels" in env.observation_spec.keys(True, True) else "observation"
        except Exception:
            self.obs_key = "observation"
        
        # Reset environment
        td = self.env.reset()
        self._last_td = td
        self.global_step = 0
        
        # Action histogram for monitoring
        self._action_hist = None
        
        # RND (intrinsic reward) - same as base trainer
        self._rnd = None
        if getattr(cfg.train, "use_intrinsic", False):
            try:
                ch = int(getattr(cfg.model, "obs_channels", 4))
                self._rnd = RND(in_channels=ch).to(device)
            except Exception:
                self._rnd = None
        
        # Configuration
        self._random_warmup_steps = int(getattr(cfg.train, "init_random_frames", 0))
        self._log_interval = int(getattr(cfg.train, "log_interval", 1000))
        self._save_every = int(getattr(cfg.train, "save_every_frames", 0))
        
        # Synthetic data tracking
        self.synthetic_generation_interval = getattr(cfg.train, 'synthetic_generation_interval', 100)
        self.last_synthetic_generation = 0
        self.synthetic_stats = {
            'total_generated': 0,
            'total_added': 0,
            'success_rate': 0.0,
        }
        
        # TD error tracking for triggers
        self.td_error_history = []
        self.recent_rewards = []
        self.exploration_bonuses = []
    
    def _append_to_replay(self, prev_td, action, next_td, llm_out: dict | None, 
                         is_synthetic: bool = False, synthetic_metadata: Optional[Dict] = None):
        """Enhanced replay append with synthetic data support."""
        try:
            from tensordict import TensorDict
        except Exception:
            from torchrl.data import TensorDict
        
        # Extract observations
        prev_obs = prev_td.get(self.obs_key)
        if prev_obs is None and ("next", self.obs_key) in prev_td.keys(True, True):
            prev_obs = prev_td.get(("next", self.obs_key))
        
        next_obs = next_td.get(("next", self.obs_key))
        if next_obs is None:
            next_obs = next_td.get(self.obs_key)
        
        # Extract reward and done
        rew = next_td.get(("next", "reward")) or next_td.get("reward")
        dn = next_td.get(("next", "done")) or next_td.get("done")
        if dn is None:
            term = next_td.get(("next", "terminated")) or next_td.get("terminated")
            trunc = next_td.get(("next", "truncated")) or next_td.get("truncated")
            if term is not None or trunc is not None:
                t = (term if term is not None else 0)
                u = (trunc if trunc is not None else 0)
                dn = (t.to(dtype=torch.bool) | u.to(dtype=torch.bool)).to(dtype=torch.float32)
            else:
                dn = None
        
        sample = TensorDict(
            {
                "observation": prev_obs.cpu(),
                "action": action.cpu(),
                "reward": (rew if rew is not None else 0.0).cpu() if hasattr(rew, "cpu") else torch.tensor(rew or 0.0),
                "done": (dn if dn is not None else torch.tensor(0.0)).cpu(),
                "next": {
                    "observation": next_obs.cpu(),
                },
            },
            batch_size=[],
        )
        
        # Add LLM annotations
        if llm_out is not None:
            if "prior_logits" in llm_out:
                sample.set("llm_prior_logits", torch.tensor(llm_out["prior_logits"]).cpu())
            if "confidence" in llm_out:
                sample.set("llm_confidence", torch.tensor(llm_out["confidence"]).cpu())
            if "mask" in llm_out:
                sample.set("llm_mask", torch.tensor(llm_out["mask"]).cpu())
            if "features" in llm_out and getattr(self.cfg.model, "llm_features_dim", 0) > 0:
                sample.set("llm_features", torch.tensor(llm_out["features"]).cpu())
        
        # Add to appropriate buffer
        if is_synthetic and synthetic_metadata:
            success = self.replay.add_synthetic(
                sample.to_dict(),
                advice_id=synthetic_metadata.get('advice_id', 'unknown'),
                llm_confidence=synthetic_metadata.get('llm_confidence', 0.5),
                execution_success=synthetic_metadata.get('execution_success', True),
                synthetic_plan=synthetic_metadata.get('synthetic_plan'),
                base_weight=synthetic_metadata.get('w_synth', 0.3),
            )
            if success:
                self.synthetic_stats['total_added'] += 1
        else:
            self.replay.add_real(sample.to_dict())
    
    def _generate_synthetic_experiences(self, current_obs: torch.Tensor) -> List[Tuple[Dict, Dict]]:
        """Generate synthetic experiences using LLM and world model."""
        if not self.synthetic_generator:
            return []
        
        # Calculate triggers
        recent_reward = np.mean(self.recent_rewards[-10:]) if self.recent_rewards else 0.0
        exploration_bonus = np.mean(self.exploration_bonuses[-10:]) if self.exploration_bonuses else 0.0
        td_error = np.mean(self.td_error_history[-10:]) if self.td_error_history else 0.0
        
        # Build context
        context = {
            'step': self.global_step,
            'recent_reward': recent_reward,
            'exploration_bonus': exploration_bonus,
            'td_error': td_error,
        }
        
        # Generate synthetic experiences
        synthetic_experiences = self.synthetic_generator.generate_synthetic_experience(
            obs=current_obs,
            context=context,
            num_actions=getattr(self.agent, "n_actions", 17),
            recent_reward=recent_reward,
            exploration_bonus=exploration_bonus,
            td_error=td_error,
        )
        
        if synthetic_experiences:
            self.synthetic_stats['total_generated'] += len(synthetic_experiences)
        
        return synthetic_experiences
    
    def _collect(self, steps: int):
        """Enhanced collection with synthetic experience generation."""
        td = self._last_td
        
        # Episode tracking
        if not hasattr(self, "_ep_return"):
            self._ep_return = 0.0
            self._ep_len = 0
        
        if self._action_hist is None:
            try:
                n = int(getattr(self.agent, "n_actions", 0))
                self._action_hist = torch.zeros(n, dtype=torch.long)
            except Exception:
                self._action_hist = None
        
        for step_idx in range(steps):
            obs = self._get_obs_tensor(td)
            
            # LLM inference with enhanced adapter
            llm_out = None
            try:
                if getattr(self.cfg.train, "use_llm", False):
                    obs_np = obs.detach().cpu().numpy()
                    
                    # Check if we should call LLM based on triggers
                    recent_reward = np.mean(self.recent_rewards[-5:]) if self.recent_rewards else 0.0
                    td_error = np.mean(self.td_error_history[-5:]) if self.td_error_history else 0.0
                    
                    context = {'step': self.global_step, 'episode_step': self._ep_len}
                    
                    if self.llm.should_call_llm(obs_np, td_error, recent_reward):
                        llm_result = self.llm.infer(obs_np, context, getattr(self.agent, "n_actions", 0))
                        if llm_result:
                            # Convert CodeOut to dict format
                            llm_out = {
                                'prior_logits': getattr(llm_result, 'policy', {}).get('logits', []),
                                'confidence': [getattr(llm_result, 'confidence', 0.5)],
                                'mask': getattr(llm_result, 'mask', []),
                                'features': getattr(llm_result, 'features', []),
                            }
            except Exception:
                llm_out = None
            
            # Action selection
            with torch.no_grad():
                if self.global_step < self._random_warmup_steps:
                    n = int(getattr(self.agent, "n_actions", 1))
                    action = torch.randint(low=0, high=n, size=(1, 1), device=self.device)
                else:
                    action = self.agent.act(obs)
            
            # Environment step
            td.set("action", action.cpu())
            prev_td = td.clone()
            td = self.env.step(td)
            
            # Extract reward
            try:
                r = float(td.get("reward").mean().item()) if "reward" in td.keys(True, True) else 0.0
            except Exception:
                r = 0.0
            
            # Intrinsic reward (same as base trainer)
            if self._rnd is not None:
                try:
                    obs_i = self._get_obs_tensor(td)
                    ri = self._rnd.intrinsic_reward(obs_i).mean().item()
                    if getattr(self.cfg.train, "intrinsic_norm", True):
                        if not hasattr(self, "_ri_mean"):
                            self._ri_mean = 0.0
                            self._ri_var = 1.0
                        m = 0.99
                        self._ri_mean = m * self._ri_mean + (1 - m) * ri
                        self._ri_var = m * self._ri_var + (1 - m) * (ri - self._ri_mean) ** 2
                        ri_n = (ri - self._ri_mean) / (self._ri_var ** 0.5 + 1e-6)
                    else:
                        ri_n = ri
                    r += float(self.cfg.train.intrinsic_coef) * ri_n
                    _ = self._rnd.update(obs_i)
                    
                    # Track exploration bonus
                    self.exploration_bonuses.append(ri_n)
                    if len(self.exploration_bonuses) > 100:
                        self.exploration_bonuses.pop(0)
                except Exception:
                    pass
            
            # Track metrics for synthetic generation triggers
            self.recent_rewards.append(r)
            if len(self.recent_rewards) > 20:
                self.recent_rewards.pop(0)
            
            # Episode tracking
            self._ep_return += r
            self._ep_len += 1
            
            if self._action_hist is not None:
                try:
                    a = int(action.squeeze().item())
                    if 0 <= a < self._action_hist.numel():
                        self._action_hist[a] += 1
                except Exception:
                    pass
            
            # Episode termination
            try:
                is_done = bool(td.get("done").item()) if "done" in td.keys(True, True) else False
            except Exception:
                is_done = False
            
            if is_done:
                # Episode logging
                try:
                    self.logger.add_scalar("env/episode_return", self._ep_return, self.global_step)
                    self.logger.add_scalar("env/episode_length", float(self._ep_len), self.global_step)
                    self.logger.add_scalar("env/episode_success", 1.0 if self._ep_return > 0.0 else 0.0, self.global_step)
                    
                    if self._action_hist is not None:
                        total = int(self._action_hist.sum().item()) or 1
                        dist = {f"a{idx}": (int(v.item()) / total) for idx, v in enumerate(self._action_hist)}
                        self.logger.add_scalars("policy/action_dist", dist, self.global_step)
                except Exception:
                    pass
                
                self._ep_return = 0.0
                self._ep_len = 0
                if self._action_hist is not None:
                    self._action_hist.zero_()
                td = self.env.reset()
            
            # Add real transition to replay
            self._append_to_replay(prev_td, action, td, llm_out, is_synthetic=False)
            
            # Generate synthetic experiences periodically
            if (self.synthetic_generator and 
                self.global_step > self._random_warmup_steps and
                (self.global_step - self.last_synthetic_generation) >= self.synthetic_generation_interval):
                
                synthetic_experiences = self._generate_synthetic_experiences(obs)
                for transition_data, metadata in synthetic_experiences:
                    # Convert to TensorDict format for replay
                    try:
                        from tensordict import TensorDict
                    except ImportError:
                        from torchrl.data import TensorDict
                    
                    # Create synthetic transition TensorDict
                    synthetic_td = TensorDict(transition_data, batch_size=[])
                    
                    # Add to replay as synthetic
                    self._append_to_replay(
                        prev_td=synthetic_td,  # Use synthetic as prev 
                        action=synthetic_td.get("action"),
                        next_td=synthetic_td,  # Use synthetic as next
                        llm_out=llm_out,
                        is_synthetic=True,
                        synthetic_metadata=metadata.to_dict(),
                    )
                
                self.last_synthetic_generation = self.global_step
            
            self._last_td = td
            self.global_step += 1
    
    def _update(self, updates: int):
        """Enhanced update with synthetic data handling."""
        logs = {}
        
        for _ in range(updates):
            try:
                # Sample batch with synthetic data support
                batch = self.replay.sample(self.cfg.train.batch_size)
            except Exception:
                break
            
            # Move to device
            for k in ["observation", "action", "reward", "done"]:
                if k in batch.keys(True, True):
                    batch.set(k, batch.get(k).to(self.device))
            
            # Move LLM data to device
            for k in batch.keys(True, True):
                if k.startswith('llm_') or k.startswith('_'):
                    batch.set(k, batch.get(k).to(self.device))
            
            # Agent update
            out = self.agent.update(batch)
            logs.update(out)
            
            # Track TD errors for synthetic generation
            if 'loss/value' in out:
                td_error = abs(out['loss/value'])
                self.td_error_history.append(td_error)
                if len(self.td_error_history) > 50:
                    self.td_error_history.pop(0)
        
        return logs
    
    def train(self):
        """Enhanced training loop with synthetic data monitoring."""
        total_frames = int(self.cfg.train.total_frames)
        collect_per_iter = int(self.cfg.train.collect_steps_per_iter)
        updates_per_collect = int(self.cfg.train.updates_per_collect)
        
        while self.global_step < total_frames:
            self._collect(collect_per_iter)
            logs = self._update(updates_per_collect)
            
            # Standard logging
            if logs:
                for k, v in logs.items():
                    if self.global_step % self._log_interval == 0:
                        self.logger.add_scalar(k, float(v), self.global_step)
            
            # Enhanced logging for synthetic data
            if self.global_step % self._log_interval == 0:
                try:
                    # Replay buffer statistics
                    replay_stats = self.replay.get_statistics()
                    for k, v in replay_stats.items():
                        self.logger.add_scalar(f"replay/{k}", v, self.global_step)
                    
                    # Synthetic generation statistics
                    if self.synthetic_generator:
                        synth_stats = self.synthetic_generator.get_statistics()
                        for k, v in synth_stats.items():
                            self.logger.add_scalar(f"synthetic_gen/{k}", v, self.global_step)
                    
                    # LLM adapter statistics
                    llm_stats = self.llm.get_metrics()
                    for k, v in llm_stats.items():
                        self.logger.add_scalar(f"llm_adapter/{k}", v, self.global_step)
                    
                    # Overall synthetic statistics
                    for k, v in self.synthetic_stats.items():
                        self.logger.add_scalar(f"synthetic_overall/{k}", v, self.global_step)
                    
                except Exception:
                    pass
            
            # Checkpointing
            try:
                if self._save_every and (self.global_step % self._save_every == 0):
                    if hasattr(self.agent, "save"):
                        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
                        path = f"{self.cfg.ckpt_dir}/ckpt_step_{self.global_step}.pt"
                        self.agent.save(path)
            except Exception:
                pass
            
            self.logger.flush()
            
            # Periodic cleanup of old synthetic data
            if self.global_step % 50000 == 0:
                removed = self.replay.clear_old_synthetic()
                if removed > 0:
                    print(f"Cleared {removed} old synthetic transitions")