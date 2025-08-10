"""LLM-guided synthetic experience generation for LoRe replay augmentation."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch

from .synthetic_replay import SyntheticTransition


class SyntheticPlan:
    """Represents a plan/macro-action from LLM for synthetic experience generation."""
    
    def __init__(
        self,
        plan_id: str,
        actions: List[int],
        expected_rewards: List[float],
        confidence: float,
        max_steps: int = 10,
        description: str = "",
    ):
        self.plan_id = plan_id
        self.actions = actions
        self.expected_rewards = expected_rewards
        self.confidence = confidence
        self.max_steps = max_steps
        self.description = description
        
        # Execution tracking
        self.executed_steps = 0
        self.actual_rewards = []
        self.success = False
        self.total_reward = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "actions": self.actions,
            "expected_rewards": self.expected_rewards,
            "confidence": self.confidence,
            "max_steps": self.max_steps,
            "description": self.description,
            "executed_steps": self.executed_steps,
            "actual_rewards": self.actual_rewards,
            "success": self.success,
            "total_reward": self.total_reward,
        }


class SyntheticExperienceGenerator:
    """Generate synthetic experiences using LLM-guided plans and world model rollouts."""
    
    def __init__(
        self,
        world_model: torch.nn.Module,
        llm_adapter,
        device: torch.device,
        max_rollout_length: int = 5,
        confidence_threshold: float = 0.3,
        success_reward_threshold: float = 0.1,
        synthetic_execution_prob: float = 0.2,  # Probability of executing LLM plans
    ):
        self.world_model = world_model
        self.llm_adapter = llm_adapter
        self.device = device
        self.max_rollout_length = max_rollout_length
        self.confidence_threshold = confidence_threshold
        self.success_reward_threshold = success_reward_threshold
        self.synthetic_execution_prob = synthetic_execution_prob
        
        # Plan cache to avoid redundant LLM calls
        self.plan_cache: Dict[str, SyntheticPlan] = {}
        self.execution_history: List[SyntheticPlan] = []
        
        # Statistics
        self.stats = {
            'plans_generated': 0,
            'plans_executed': 0,
            'plans_successful': 0,
            'avg_plan_confidence': 0.0,
            'avg_execution_reward': 0.0,
            'cache_hit_rate': 0.0,
        }
        
        self.cache_hits = 0
        self.cache_misses = 0
    
    def should_generate_synthetic(
        self,
        current_obs: torch.Tensor,
        recent_reward: float = 0.0,
        exploration_bonus: float = 0.0,
        td_error: float = 0.0,
    ) -> bool:
        """Determine if synthetic experience should be generated based on LoRe triggers."""
        
        # Low reward situations (struggling)
        if recent_reward < -0.1:
            return np.random.random() < self.synthetic_execution_prob * 2.0
        
        # High TD error (learning opportunity)
        if td_error > 0.3:
            return np.random.random() < self.synthetic_execution_prob * 1.5
        
        # High exploration bonus (novel states)
        if exploration_bonus > 0.2:
            return np.random.random() < self.synthetic_execution_prob * 1.8
        
        # Regular probability
        return np.random.random() < self.synthetic_execution_prob
    
    def _compute_state_hash(self, obs: torch.Tensor) -> str:
        """Compute hash for state-based plan caching."""
        # Downsample observation for stable hashing
        if obs.dim() == 4:  # [1, C, H, W]
            obs_small = torch.nn.functional.avg_pool2d(obs, kernel_size=8)
        else:
            obs_small = obs
        
        # Quantize to reduce hash collisions from small variations
        obs_quantized = torch.round(obs_small * 8.0) / 8.0
        obs_bytes = obs_quantized.detach().cpu().numpy().tobytes()
        return hashlib.md5(obs_bytes).hexdigest()[:12]
    
    def generate_plan_with_llm(
        self,
        obs: torch.Tensor,
        context: Dict[str, Any],
        num_actions: int,
    ) -> Optional[SyntheticPlan]:
        """Generate plan using LLM adapter."""
        
        # Check cache first
        state_hash = self._compute_state_hash(obs)
        if state_hash in self.plan_cache:
            self.cache_hits += 1
            cached_plan = self.plan_cache[state_hash]
            # Return copy with new ID for tracking
            new_plan = SyntheticPlan(
                plan_id=str(uuid.uuid4())[:8],
                actions=cached_plan.actions.copy(),
                expected_rewards=cached_plan.expected_rewards.copy(),
                confidence=cached_plan.confidence * 0.9,  # Slight confidence decay for cached
                max_steps=cached_plan.max_steps,
                description=f"cached: {cached_plan.description}",
            )
            return new_plan
        
        self.cache_misses += 1
        
        try:
            # Enhanced prompt for plan generation
            obs_np = obs.detach().cpu().numpy()
            
            # Build context for LLM
            enhanced_context = context.copy()
            enhanced_context.update({
                'obs_shape': list(obs.shape),
                'request_type': 'synthetic_plan',
                'max_steps': self.max_rollout_length,
                'available_actions': list(range(num_actions)),
            })
            
            # Try enhanced adapter first, fallback to basic
            if hasattr(self.llm_adapter, 'infer_batch'):
                results = self.llm_adapter.infer_batch([obs_np], [enhanced_context], num_actions)
                llm_result = results[0] if results and results[0] is not None else None
            else:
                llm_result = self.llm_adapter.infer(obs_np, num_actions)
            
            if not llm_result:
                return None
            
            # Parse LLM output for plan generation
            plan = self._parse_llm_plan_output(llm_result, num_actions)
            
            if plan and plan.confidence > self.confidence_threshold:
                # Cache successful plan
                self.plan_cache[state_hash] = plan
                
                # Limit cache size
                if len(self.plan_cache) > 200:
                    oldest_key = next(iter(self.plan_cache))
                    del self.plan_cache[oldest_key]
                
                self.stats['plans_generated'] += 1
                return plan
        
        except Exception:
            pass  # Fail silently for robustness
        
        return None
    
    def _parse_llm_plan_output(
        self, 
        llm_output: Dict[str, Any], 
        num_actions: int
    ) -> Optional[SyntheticPlan]:
        """Parse LLM output to extract plan information."""
        try:
            # Extract plan from different possible formats
            actions = []
            expected_rewards = []
            confidence = 0.5
            description = "llm_plan"
            
            # Try to extract from 'policy' field (action sequence)
            if isinstance(llm_output, dict):
                # Enhanced adapter format
                if hasattr(llm_output, 'policy') and llm_output.policy:
                    if 'action_sequence' in llm_output.policy:
                        actions = llm_output.policy['action_sequence'][:self.max_rollout_length]
                    elif 'logits' in llm_output.policy:
                        # Convert logits to action sequence
                        logits = llm_output.policy['logits']
                        if len(logits) == num_actions:
                            # Use top actions as sequence
                            action_probs = torch.softmax(torch.tensor(logits), dim=0)
                            top_actions = torch.topk(action_probs, k=min(3, self.max_rollout_length)).indices.tolist()
                            actions = top_actions
                
                # Extract confidence
                confidence = float(getattr(llm_output, 'confidence', 0.5))
                description = str(getattr(llm_output, 'notes', 'llm_generated_plan'))
                
                # Extract expected rewards if available
                if hasattr(llm_output, 'subgoal') and llm_output.subgoal:
                    expected_rewards = [0.1] * len(actions)  # Optimistic assumption
                else:
                    expected_rewards = [0.05] * len(actions)  # Conservative assumption
                
            else:
                # Basic adapter format
                if 'prior_logits' in llm_output:
                    logits = llm_output['prior_logits']
                    if len(logits) == num_actions:
                        # Use top actions as sequence
                        action_probs = torch.softmax(torch.tensor(logits), dim=0)
                        top_actions = torch.topk(action_probs, k=min(3, self.max_rollout_length)).indices.tolist()
                        actions = top_actions
                
                confidence = float(llm_output.get('confidence', [0.5])[0])
                expected_rewards = [0.05] * len(actions)
            
            # Validate actions
            if not actions or not all(0 <= a < num_actions for a in actions):
                # Fallback: generate reasonable action sequence
                actions = [np.random.randint(0, num_actions) for _ in range(min(3, self.max_rollout_length))]
                expected_rewards = [0.02] * len(actions)
                confidence *= 0.5  # Reduce confidence for fallback
            
            return SyntheticPlan(
                plan_id=str(uuid.uuid4())[:8],
                actions=actions,
                expected_rewards=expected_rewards,
                confidence=confidence,
                max_steps=len(actions),
                description=description,
            )
        
        except Exception:
            return None
    
    def execute_plan_with_world_model(
        self,
        plan: SyntheticPlan,
        initial_obs: torch.Tensor,
        initial_hidden: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Execute plan using world model to generate synthetic transitions."""
        
        synthetic_transitions = []
        
        try:
            # Encode initial observation
            with torch.no_grad():
                if hasattr(self.world_model, 'encoder'):
                    current_obs = initial_obs
                    if initial_hidden is not None:
                        current_hidden = initial_hidden
                    else:
                        current_hidden = torch.zeros(1, initial_obs.size(0), 256, device=self.device)
                
                for step, action_idx in enumerate(plan.actions):
                    if step >= plan.max_steps:
                        break
                    
                    # Convert action to tensor
                    action = torch.tensor([action_idx], device=self.device).unsqueeze(0)
                    
                    # World model forward pass
                    if hasattr(self.world_model, 'encoder') and hasattr(self.world_model, 'rssm'):
                        # Encode current observation
                        obs_encoded = self.world_model.encoder(current_obs)
                        
                        # RSSM step with action
                        next_hidden, _ = self.world_model.rssm(obs_encoded.unsqueeze(1), current_hidden)
                        
                        # Decode next observation
                        if hasattr(self.world_model, 'decoder'):
                            next_obs = self.world_model.decoder(next_hidden.squeeze(1))
                        else:
                            # Fallback: add noise to current observation
                            next_obs = current_obs + torch.randn_like(current_obs) * 0.05
                        
                        # Predict reward
                        if hasattr(self.world_model, 'reward_head'):
                            predicted_reward = self.world_model.reward_head(next_hidden.squeeze(1))
                            reward = float(predicted_reward.mean().item())
                        else:
                            reward = plan.expected_rewards[step] if step < len(plan.expected_rewards) else 0.0
                    
                    else:
                        # Fallback for simpler world model
                        next_obs = current_obs + torch.randn_like(current_obs) * 0.03
                        next_hidden = current_hidden
                        reward = plan.expected_rewards[step] if step < len(plan.expected_rewards) else 0.0
                    
                    # Create synthetic transition
                    transition = {
                        'observation': current_obs.cpu(),
                        'action': action.cpu(),
                        'reward': torch.tensor(reward),
                        'done': torch.tensor(False),  # Most synthetic transitions are not terminal
                        'next': {
                            'observation': next_obs.cpu(),
                        }
                    }
                    
                    synthetic_transitions.append(transition)
                    plan.executed_steps += 1
                    plan.actual_rewards.append(reward)
                    plan.total_reward += reward
                    
                    # Update for next step
                    current_obs = next_obs
                    current_hidden = next_hidden
                
                # Determine plan success
                plan.success = (
                    plan.total_reward > self.success_reward_threshold and
                    plan.executed_steps >= len(plan.actions) * 0.7  # At least 70% completed
                )
                
                self.stats['plans_executed'] += 1
                if plan.success:
                    self.stats['plans_successful'] += 1
                
                self.execution_history.append(plan)
                if len(self.execution_history) > 100:
                    self.execution_history.pop(0)
        
        except Exception:
            plan.success = False  # Mark as failed on exception
        
        return synthetic_transitions
    
    def generate_synthetic_experience(
        self,
        obs: torch.Tensor,
        context: Dict[str, Any],
        num_actions: int,
        recent_reward: float = 0.0,
        exploration_bonus: float = 0.0,
        td_error: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], SyntheticTransition]]:
        """Main method to generate synthetic experience following LoRe specification."""
        
        # Check if we should generate synthetic experience
        if not self.should_generate_synthetic(obs, recent_reward, exploration_bonus, td_error):
            return []
        
        # Generate plan with LLM
        plan = self.generate_plan_with_llm(obs, context, num_actions)
        if not plan:
            return []
        
        # Execute plan with world model
        synthetic_transitions = self.execute_plan_with_world_model(plan, obs)
        if not synthetic_transitions:
            return []
        
        # Create synthetic metadata for each transition
        results = []
        base_weight = min(0.8, plan.confidence * 0.6)  # Base weight from confidence
        
        for i, transition in enumerate(synthetic_transitions):
            # Weight decays with position in sequence (later steps less reliable)
            position_decay = 0.9 ** i
            final_weight = base_weight * position_decay
            
            metadata = SyntheticTransition(
                is_synth=True,
                w_synth=final_weight,
                advice_id=plan.plan_id,
                origin="llm",
                llm_confidence=plan.confidence,
                execution_success=plan.success,
                synthetic_plan=plan.to_dict(),
            )
            
            results.append((transition, metadata))
        
        self._update_stats()
        return results
    
    def _update_stats(self) -> None:
        """Update generator statistics."""
        if self.execution_history:
            confidences = [p.confidence for p in self.execution_history[-50:]]
            rewards = [p.total_reward for p in self.execution_history[-50:]]
            
            self.stats['avg_plan_confidence'] = np.mean(confidences)
            self.stats['avg_execution_reward'] = np.mean(rewards)
        
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.stats['cache_hit_rate'] = self.cache_hits / total_cache_requests
    
    def get_statistics(self) -> Dict[str, float]:
        """Get generator statistics for monitoring."""
        return self.stats.copy()
    
    def clear_cache(self) -> None:
        """Clear plan cache and reset statistics."""
        self.plan_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0