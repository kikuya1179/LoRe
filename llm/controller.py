from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .enhanced_adapter import EnhancedLLMAdapter, LLMAdapterConfigV2


@dataclass
class LLMControllerConfig:
    budget_total: int = 200
    cooldown_steps: int = 200
    success_cooldown_steps: int = 500
    novelty_threshold: float = 0.1
    td_error_threshold: float = 0.2
    plateau_frames: int = 1000


class LLMController:
    """Budgeted LLM controller with cooldown and simple caching."""

    def __init__(self, cfg: Any, num_actions: int, lore_cfg: Any) -> None:
        self.cfg = cfg
        self.num_actions = num_actions
        self.lore_cfg = lore_cfg

        adapter_cfg = LLMAdapterConfigV2(
            enabled=True,
            timeout_s=2.5,
            api_retries=2,
            use_dsl=False,
            batch_size=4,
            cache_size=1000,
            novelty_threshold=getattr(cfg, 'novelty_threshold', 0.1),
            td_error_threshold=getattr(cfg, 'td_error_threshold', 0.2),
            plateau_frames=getattr(cfg, 'plateau_frames', 1000),
        )
        self.llm_adapter = EnhancedLLMAdapter(num_actions=num_actions, cfg=adapter_cfg)

        # Budget and cooldown
        self.budget_remaining = int(getattr(cfg, 'budget_total', 200))
        self.cooldown_remaining = 0
        self.last_success_step = -10**9

        # Cache (simple dict)
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def to(self, device: torch.device) -> None:
        # Placeholder to keep interface symmetry
        return

    def step(self) -> None:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def _key(self, obs_np: np.ndarray, context: Dict[str, Any]) -> str:
        shape = tuple(obs_np.shape)
        ctx = (context.get('mission', ''), context.get('remaining_steps', 0))
        return str((shape, ctx))

    def request(self, obs_np: np.ndarray, context: Dict[str, Any], reward: float, done: bool,
                td_error: float, step: int) -> Optional[Any]:
        # Cooldown budget checks
        if self.budget_remaining <= 0:
            return None
        if self.cooldown_remaining > 0:
            return None

        # Simple novelty using cache-miss
        key = self._key(obs_np, context)
        novelty = 0.0 if key in self.cache else 1.0

        # Initialize missing EMA trackers lazily for backward compatibility
        if not hasattr(self, 'td_error_ema'):
            self.td_error_ema = 0.0
        if not hasattr(self, 'td_error_ema_alpha'):
            self.td_error_ema_alpha = 0.05
        # Trigger conditions
        should_call = self.llm_adapter.should_call_llm(obs_np, td_error=td_error, performance=reward, novelty=novelty)
        if not should_call:
            return None

        # Call adapter
        result = self.llm_adapter.get_action_logits(obs_np, context)
        if result is not None:
            # Cache and bookkeeping
            if key in self.cache:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            self.cache[key] = result

            self.budget_remaining -= 1
            self.cooldown_remaining = int(getattr(self.cfg, 'cooldown_steps', 200))

        return result

    def get_statistics(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) if total > 0 else 0.0
        return {
            'llm_budget_remaining': self.budget_remaining,
            'llm_cooldown_remaining': self.cooldown_remaining,
            'llm_calls_used': getattr(self.cfg, 'budget_total', 200) - self.budget_remaining,
            'llm_cache_hit_rate': hit_rate,
        }

"""
LLM Controller for LoRe Implementation
Manages budget, cooldown, triggers, and caching for LLM requests
"""

import time
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from collections import deque, defaultdict
import numpy as np

from ..conf import LLMConfig, LoReConfig
from ..utils.code_out import CodeOut
from .enhanced_adapter import EnhancedLLMAdapter
from .priornet import PriorNetDistiller


class LLMController:
    """
    Controls when and how LLM requests are made based on:
    - Budget constraints
    - Cooldown periods
    - Trigger conditions (plateau, novelty, TD error)
    - Response caching
    """
    
    def __init__(self, config: LLMConfig, num_actions: int, lore_config: LoReConfig = None):
        self.config = config
        self.lore_config = lore_config or LoReConfig()
        self.num_actions = num_actions
        
        # Budget tracking
        self.calls_used = 0
        self.total_budget = config.budget_total
        
        # Cooldown tracking
        self.cooldown_remaining = 0
        self.last_success_step = -1
        
        # Trigger state
        self.reward_history = deque(maxlen=config.plateau_frames)
        self.success_history = deque(maxlen=config.plateau_frames)
        # TD error z-score trackers (EMA mean/std)
        self.td_mean = 0.0
        self.td_var = 1e-6
        self.td_alpha = 0.01
        
        # Novelty tracking (simplified state hash counting)
        self.state_counts = defaultdict(int)
        self.max_cache_size = 10000
        
        # Response cache with TTL and local cooldown per key
        # key -> {response, ts, hits, local_cd}
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.trigger_counts = {
            'budget': 0,
            'cooldown': 0, 
            'plateau': 0,
            'novelty': 0,
            'td_error': 0,
            'cache_hit': 0,
            'invalid_response': 0,
            'priornet_used': 0
        }

        # TD error EMA trackers (for trigger logic)
        self.td_error_ema: float = 0.0
        self.td_error_ema_alpha: float = getattr(self.config, 'td_error_ema_alpha', 0.05)

        # Additional guard params
        self.beta_min = getattr(self.lore_config, 'beta_min', 0.1)
        # Hysteresis thresholds
        self.tau_low = getattr(self.lore_config, 'hysteresis_tau_low', 0.4)
        self.tau_high = getattr(self.lore_config, 'hysteresis_tau_high', 0.6)
        self.trigger_state = {'plateau': False, 'novelty': False, 'td': False}

        # Cache TTL / local cooldown
        self.cache_ttl_steps = getattr(config, 'cache_ttl_steps', 500)
        self.cache_ttl_hits = getattr(config, 'cache_ttl_hits', 3)
        self.key_local_cooldown_steps = getattr(config, 'key_local_cooldown_steps', 50)

        # Novelty adaptive probability
        self.novelty_p_max = getattr(config, 'novelty_p_max', 0.3)
        self.novelty_a = getattr(config, 'novelty_a', 0.2)

        # Plateau thresholds
        self.plateau_w = getattr(config, 'plateau_frames', 1000)
        self.plateau_tau = getattr(config, 'plateau_tau', 0.05)  # relative to reward range
        self.plateau_grad_thresh = getattr(config, 'plateau_grad_thresh', 1e-3)

        # Uplift/backoff
        self.uplift_horizon = getattr(config, 'uplift_horizon', 50)
        self.backoff_M = getattr(config, 'backoff_M', 3)
        self.backoff_cooldown_max = getattr(config, 'backoff_cooldown_max', 2000)
        self.cooldown_extra_success = getattr(config, 'success_cooldown_steps', 500)
        self._reward_cumsum = 0.0
        self._reward_cumsum_history: Dict[int, float] = {}
        self._pending_calls: deque[Tuple[int, float]] = deque(maxlen=100)
        self._ineffective_streak = 0
        
        # LLM adapter and PriorNet distiller
        self.llm_adapter = EnhancedLLMAdapter(num_actions)
        self.priornet_distiller = None
        if self.lore_config.use_priornet:
            self.priornet_distiller = PriorNetDistiller(
                self.lore_config, 
                obs_channels=1,  # Will be updated when first observation is seen
                n_actions=num_actions
            )
        
    def step(self):
        """Called each environment step to update cooldown"""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
    
    def request(self, obs_np: np.ndarray, context: Dict[str, Any], 
                reward: float, done: bool, td_error: float, 
                step: int, beta_est: float | None = None) -> Optional[CodeOut]:
        """
        Main request method that checks all conditions and triggers
        
        Args:
            obs_np: Current observation as numpy array
            context: Environment context (mission, remaining steps, etc.)
            reward: Current reward
            done: Episode done flag
            td_error: TD error for current state
            step: Current training step
            
        Returns:
            CodeOut if LLM should be queried, None otherwise
        """
        # Update internal state
        self.reward_history.append(reward)
        self.success_history.append(1.0 if done and reward > 0 else 0.0)
        self.td_error_ema = (1 - self.td_error_ema_alpha) * self.td_error_ema + \
                            self.td_error_ema_alpha * abs(td_error)
        
        if done and reward > 0:
            self.last_success_step = step
        
        # Check budget constraint
        if self.calls_used >= self.total_budget:
            self.trigger_counts['budget'] += 1
            return None
            
        # Check cooldown
        if self.cooldown_remaining > 0:
            self.trigger_counts['cooldown'] += 1
            # allow cache bypass if key-local cooldown is zero and TTL valid
            state_hash = self._hash_state(obs_np, context)
            cached = self._get_cache_if_valid(state_hash, step, allow_bypass=True)
            if cached is not None:
                self.trigger_counts['cache_hit'] += 1
                return cached
            return None
            
        # Generate state hash for noveltyとキャッシュ
        state_hash = self._hash_state(obs_np, context)
        
        # Cache fast-path with TTL and local cooldown
        cached = self._get_cache_if_valid(state_hash, step)
        if cached is not None:
            self.trigger_counts['cache_hit'] += 1
            return cached
        
        # β gate: low-uncertainty states skip
        if beta_est is not None and beta_est < self.beta_min:
            return None

        # Try PriorNet first if available and ready
        if (self.priornet_distiller and 
            self.priornet_distiller.should_use_priornet() and
            self._should_trigger(state_hash, step, td_error)):
            
            priornet_response = self.priornet_distiller.predict(obs_np, context)
            if priornet_response is not None:
                self.trigger_counts['priornet_used'] += 1
                
                # Convert to CodeOut format
                response = CodeOut(
                    features=priornet_response.get('features', []),
                    subgoal=None,
                    r_shaped=0.0,
                    policy=priornet_response.get('policy', {}),
                    mask=priornet_response.get('mask', [1] * self.num_actions),
                    confidence=priornet_response.get('confidence', 0.8),
                    notes=priornet_response.get('notes', 'PriorNet prediction')
                )
                
                # Cache PriorNet response too
                if len(self.response_cache) < self.max_cache_size:
                    self.response_cache[state_hash] = response
                
                return response
            
        # Check trigger conditions for LLM call (with hysteresis)
        if not self._should_trigger(state_hash, step, td_error):
            return None
            
        # Make LLM request
        try:
            response = self.llm_adapter.get_action_logits(obs_np, context)
            
            if response is None:
                self.trigger_counts['invalid_response'] += 1
                return None
                
            # Update tracking
            self.calls_used += 1
            self.cache_misses += 1
            
            # Add to PriorNet training data if distiller is available
            if (self.priornet_distiller and response.policy and 
                'logits' in response.policy):
                self.priornet_distiller.add_llm_sample(
                    obs=obs_np,
                    context=context,
                    llm_logits=np.array(response.policy['logits']),
                    llm_features=response.features if response.features else [],
                    confidence=response.confidence if hasattr(response, 'confidence') else 1.0
                )
            
            # Cache response with TTL and local cooldown reset
            if len(self.response_cache) < self.max_cache_size:
                self.response_cache[state_hash] = {
                    'response': response,
                    'ts': step,
                    'hits': 0,
                    'local_cd': self.key_local_cooldown_steps,
                }
                
            # Set cooldown（成功時は延長）
            self.cooldown_remaining = self.config.cooldown_steps
            if done and reward > 0:
                self.cooldown_remaining += self.cooldown_extra_success

            # Track for uplift/backoff
            self._pending_calls.append((step, self._reward_cumsum))
                
            return response
            
        except Exception as e:
            print(f"LLM request failed: {e}")
            self.trigger_counts['invalid_response'] += 1
            return None
    
    def _should_trigger(self, state_hash: str, step: int, td_error: float) -> bool:
        """Check if any trigger conditions are met"""
        # Plateau detection（統計化）
        plateau_flag = False
        if len(self.reward_history) >= self.plateau_w:
            half = self.plateau_w // 2
            recent = list(self.reward_history)[-half:]
            older = list(self.reward_history)[:half]
            if recent and older:
                r_mean = float(np.mean(recent))
                o_mean = float(np.mean(older))
                improvement = abs(r_mean - o_mean)
                # 近似の勾配（一次差分の平均）
                grad = 0.0
                if len(recent) >= 2:
                    diffs = np.diff(recent)
                    grad = float(np.mean(diffs))
                plateau_raw = (improvement < self.plateau_tau) and (abs(grad) < self.plateau_grad_thresh)
                # Hysteresis on plateau: simple state latch
                if self.trigger_state['plateau']:
                    plateau_flag = plateau_raw or (improvement < self.plateau_tau * 1.2)
                    if not plateau_flag:
                        self.trigger_state['plateau'] = False
                else:
                    plateau_flag = plateau_raw
                    if plateau_flag:
                        self.trigger_state['plateau'] = True
        
        # Novelty detection
        self.state_counts[state_hash] += 1
        cnt = self.state_counts[state_hash]
        p = min(self.novelty_p_max, self.novelty_a / max(1, cnt))
        novelty_flag = (np.random.random() < p)
        
        # TD error spike
        # Update EMA mean/std
        delta = td_error - self.td_mean
        self.td_mean += self.td_alpha * delta
        self.td_var = (1 - self.td_alpha) * self.td_var + self.td_alpha * (td_error - self.td_mean) ** 2
        td_std = max(self.td_var ** 0.5, 1e-6)
        z = abs(td_error - self.td_mean) / td_std
        td_flag = (z > getattr(self.config, 'td_error_threshold', 2.0))

        # Hysteresis combine
        trigger = plateau_flag or novelty_flag or td_flag
        if plateau_flag:
            self.trigger_counts['plateau'] += 1
        if novelty_flag:
            self.trigger_counts['novelty'] += 1
        if td_flag:
            self.trigger_counts['td_error'] += 1
        return trigger
    
    def _hash_state(self, obs_np: np.ndarray, context: Dict[str, Any]) -> str:
        """Generate hash for state observation and context"""
        # Simple hash of observation and mission
        obs_bytes = obs_np.tobytes()
        mission = context.get('mission', '')
        remaining = context.get('remaining_steps', 0)
        
        hash_input = f"{obs_bytes.hex()}-{mission}-{remaining}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics for logging"""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(cache_total, 1)
        
        stats = {
            'llm_calls_used': self.calls_used,
            'llm_budget_remaining': self.total_budget - self.calls_used,
            'llm_cooldown_remaining': self.cooldown_remaining,
            'llm_cache_hit_rate': cache_hit_rate,
            'llm_cache_size': len(self.response_cache),
            'llm_td_error_ema': self.td_error_ema,
            **{f'llm_trigger_{k}': v for k, v in self.trigger_counts.items()}
        }
        
        # Add PriorNet statistics if available
        if self.priornet_distiller:
            priornet_stats = self.priornet_distiller.get_statistics()
            stats.update(priornet_stats)
        # Call rate & backoff metrics
        stats.update({
            'llm_call_rate_per_1k': self.calls_used / 10.0,
            'llm_key_local_cd_keys': sum(1 for v in self.response_cache.values() if v.get('local_cd', 0) > 0),
            'llm_cache_size': len(self.response_cache),
            'llm_ineffective_streak': self._ineffective_streak,
        })
        
        return stats

    def record_reward(self, step: int, reward: float, done: bool):
        """Record reward stream for plateau/uplift/backoff."""
        self.reward_history.append(float(reward))
        self._reward_cumsum += float(reward)
        self._reward_cumsum_history[step] = self._reward_cumsum
        # Evaluate pending calls after horizon
        while self._pending_calls and step - self._pending_calls[0][0] >= self.uplift_horizon:
            s0, r0 = self._pending_calls.popleft()
            uplift = self._reward_cumsum - r0
            if uplift <= 0.0:
                self._ineffective_streak += 1
                # Backoff: increase cooldown a bit
                self.cooldown_remaining = min(self.cooldown_remaining * 2 + 10, self.backoff_cooldown_max)
                # Reduce novelty aggressiveness slightly
                self.novelty_a = max(0.05, self.novelty_a * 0.9)
            else:
                self._ineffective_streak = 0

    def notify_success(self, step: int):
        """Extend cooldown on success."""
        self.cooldown_remaining += self.cooldown_extra_success

    def _get_cache_if_valid(self, state_hash: str, step: int, allow_bypass: bool = False) -> Optional[CodeOut]:
        entry = self.response_cache.get(state_hash)
        if not entry:
            return None
        # local cooldown
        if entry['local_cd'] > 0 and not allow_bypass:
            entry['local_cd'] -= 1
            return None
        # TTL
        if (step - entry['ts'] > self.cache_ttl_steps) or (entry['hits'] >= self.cache_ttl_hits):
            # expire
            self.response_cache.pop(state_hash, None)
            return None
        # valid
        entry['hits'] += 1
        self.cache_hits += 1
        resp = entry['response']
        # set local cooldown after use
        entry['local_cd'] = self.key_local_cooldown_steps
        return resp
    
    def reset_budget(self, new_budget: Optional[int] = None):
        """Reset budget for new training period"""
        self.calls_used = 0
        if new_budget is not None:
            self.total_budget = new_budget
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def to(self, device):
        """Move PriorNet to device if available"""
        if self.priornet_distiller:
            self.priornet_distiller.to(device)
        return self
    
    def save_priornet(self, path: str):
        """Save PriorNet model if available"""
        if self.priornet_distiller:
            self.priornet_distiller.save(path)
    
    def load_priornet(self, path: str):
        """Load PriorNet model if available"""
        if self.priornet_distiller:
            self.priornet_distiller.load(path)