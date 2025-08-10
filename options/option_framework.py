"""LoRe Option Framework - Hierarchical skills with LLM generation."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass 
class OptionSpec:
    """Specification for a hierarchical option/skill."""
    
    option_id: str
    name: str
    description: str
    primitive_actions: List[int]  # Sequence of primitive actions
    expected_duration: int  # Expected execution steps
    success_condition: Optional[str] = None  # Natural language success condition
    failure_condition: Optional[str] = None  # Natural language failure condition
    confidence: float = 0.5  # LLM confidence in this option
    generation_context: Dict[str, Any] = None  # Context when generated
    
    # Execution statistics
    total_executions: int = 0
    successful_executions: int = 0
    avg_duration: float = 0.0
    avg_reward: float = 0.0
    
    def __post_init__(self):
        if self.generation_context is None:
            self.generation_context = {}
    
    def success_rate(self) -> float:
        """Calculate success rate of this option."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    def update_stats(self, duration: int, reward: float, success: bool):
        """Update execution statistics."""
        # Exponential moving average for stability
        alpha = 0.1
        
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        
        if self.avg_duration == 0:
            self.avg_duration = duration
        else:
            self.avg_duration = (1 - alpha) * self.avg_duration + alpha * duration
        
        if self.avg_reward == 0:
            self.avg_reward = reward
        else:
            self.avg_reward = (1 - alpha) * self.avg_reward + alpha * reward
    
    def is_viable(self, min_success_rate: float = 0.1, min_executions: int = 3) -> bool:
        """Check if option is viable enough to keep."""
        if self.total_executions < min_executions:
            return True  # Give new options a chance
        return self.success_rate() >= min_success_rate


class OptionExecution:
    """Tracks the execution state of an active option."""
    
    def __init__(self, option_spec: OptionSpec, start_step: int):
        self.option_spec = option_spec
        self.start_step = start_step
        self.current_action_idx = 0
        self.accumulated_reward = 0.0
        self.steps_executed = 0
        self.is_terminated = False
        self.termination_reason = "running"  # "running", "success", "failure", "timeout", "early_termination"
        self.max_steps = max(option_spec.expected_duration * 2, 10)  # Safety limit
    
    def get_next_action(self) -> Optional[int]:
        """Get next primitive action to execute."""
        if (self.is_terminated or 
            self.current_action_idx >= len(self.option_spec.primitive_actions)):
            return None
        
        action = self.option_spec.primitive_actions[self.current_action_idx]
        self.current_action_idx += 1
        return action
    
    def step(self, reward: float) -> bool:
        """Update execution state with reward. Returns True if option should terminate."""
        self.accumulated_reward += reward
        self.steps_executed += 1
        
        # Check timeout
        if self.steps_executed >= self.max_steps:
            self.is_terminated = True
            self.termination_reason = "timeout"
            return True
        
        # Check if all actions completed
        if self.current_action_idx >= len(self.option_spec.primitive_actions):
            self.is_terminated = True
            self.termination_reason = "success" if self.accumulated_reward > 0 else "completed"
            return True
        
        return False
    
    def force_terminate(self, reason: str = "early_termination"):
        """Force termination of option."""
        self.is_terminated = True
        self.termination_reason = reason
    
    def is_successful(self) -> bool:
        """Determine if execution was successful."""
        return (self.termination_reason in ["success", "completed"] and 
                self.accumulated_reward > -0.1)  # Small tolerance for noise


class OptionTerminationPredictor(nn.Module):
    """Neural network to predict when options should terminate."""
    
    def __init__(self, latent_dim: int, num_options: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_options = num_options
        
        # Termination predictor for each option
        self.termination_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),  # +1 for option execution time
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_options),  # One termination probability per option
            nn.Sigmoid(),
        )
        
        # Option value estimator (how good is continuing this option)
        self.option_value_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, num_options),
        )
    
    def forward(self, latent_state: torch.Tensor, option_id: int, 
                execution_time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict termination probability and option value.
        
        Args:
            latent_state: [batch_size, latent_dim]
            option_id: ID of currently executing option
            execution_time: [batch_size] number of steps option has been running
            
        Returns:
            termination_prob: [batch_size] probability of terminating
            option_value: [batch_size] value of continuing this option
        """
        batch_size = latent_state.size(0)
        
        # Prepare input with execution time
        execution_time_norm = execution_time.float().unsqueeze(-1) / 10.0  # Normalize
        termination_input = torch.cat([latent_state, execution_time_norm], dim=-1)
        
        # Predict termination probabilities for all options
        all_termination_probs = self.termination_net(termination_input)
        
        # Extract termination probability for current option
        option_indices = torch.full((batch_size,), option_id, device=latent_state.device)
        termination_prob = all_termination_probs.gather(-1, option_indices.unsqueeze(-1)).squeeze(-1)
        
        # Predict option values
        all_option_values = self.option_value_net(latent_state)
        option_value = all_option_values.gather(-1, option_indices.unsqueeze(-1)).squeeze(-1)
        
        return termination_prob, option_value


class DualHeadActor(nn.Module):
    """Actor with separate heads for primitive actions and options."""
    
    def __init__(
        self, 
        latent_dim: int, 
        num_primitive_actions: int,
        num_options: int,
        llm_features_dim: int = 0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_primitive_actions = num_primitive_actions
        self.num_options = num_options
        self.llm_features_dim = llm_features_dim
        
        # Feature processing
        if llm_features_dim > 0:
            self.feature_norm = nn.LayerNorm(llm_features_dim)
            self.feature_proj = nn.Linear(llm_features_dim, latent_dim // 4)
            actor_input_dim = latent_dim + latent_dim // 4
        else:
            self.feature_norm = None
            self.feature_proj = None
            actor_input_dim = latent_dim
        
        # Shared feature processing
        self.shared_net = nn.Sequential(
            nn.Linear(actor_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        
        # Primitive action head
        self.primitive_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, num_primitive_actions),
        )
        
        # Option head
        self.option_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, num_options),
        )
        
        # Action type selector (primitive vs option)
        self.action_type_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 2),  # [primitive_logit, option_logit]
        )
    
    def forward(
        self, 
        latent_state: torch.Tensor, 
        llm_features: Optional[torch.Tensor] = None,
        available_options: Optional[torch.Tensor] = None,  # [batch, num_options] mask
    ) -> Dict[str, torch.Tensor]:
        """Forward pass producing primitive and option logits.
        
        Returns:
            dict with keys:
            - 'action_type_logits': [batch, 2] for primitive vs option choice
            - 'primitive_logits': [batch, num_primitive_actions]
            - 'option_logits': [batch, num_options]
            - 'combined_logits': [batch, num_primitive_actions + num_options]
        """
        # Feature concatenation (same as original actor)
        actor_input = latent_state
        if self.feature_proj is not None:
            if llm_features is not None:
                llm_feat_norm = self.feature_norm(llm_features)
                llm_feat_proj = self.feature_proj(llm_feat_norm)
            else:
                llm_feat_proj = torch.zeros(
                    latent_state.size(0), self.feature_proj.out_features, 
                    device=latent_state.device, dtype=latent_state.dtype
                )
            actor_input = torch.cat([latent_state, llm_feat_proj], dim=-1)
        
        # Shared processing
        shared_features = self.shared_net(actor_input)
        
        # Compute logits for each head
        action_type_logits = self.action_type_head(shared_features)
        primitive_logits = self.primitive_head(shared_features)
        option_logits = self.option_head(shared_features)
        
        # Apply option availability mask
        if available_options is not None:
            option_mask = available_options.float()
            option_logits = option_logits + (1 - option_mask) * (-1e9)
        
        # Combine into single action space for compatibility
        # Format: [primitive_actions..., options...]
        combined_logits = torch.cat([primitive_logits, option_logits], dim=-1)
        
        return {
            'action_type_logits': action_type_logits,
            'primitive_logits': primitive_logits,
            'option_logits': option_logits,
            'combined_logits': combined_logits,
        }
    
    def sample_action(
        self, 
        latent_state: torch.Tensor,
        llm_features: Optional[torch.Tensor] = None,
        available_options: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[int, bool, Dict[str, torch.Tensor]]:
        """Sample action from the dual-head policy.
        
        Returns:
            action_id: int (primitive action or option id + num_primitives)
            is_option: bool (True if option was selected)
            info: dict with logits and probabilities
        """
        with torch.no_grad():
            outputs = self.forward(latent_state, llm_features, available_options)
            
            # Sample action type (primitive vs option)
            action_type_probs = F.softmax(outputs['action_type_logits'] / temperature, dim=-1)
            action_type = torch.multinomial(action_type_probs, 1).item()
            
            if action_type == 0:  # Primitive action
                primitive_probs = F.softmax(outputs['primitive_logits'] / temperature, dim=-1)
                action_id = torch.multinomial(primitive_probs, 1).item()
                is_option = False
            else:  # Option
                option_probs = F.softmax(outputs['option_logits'] / temperature, dim=-1)
                option_id = torch.multinomial(option_probs, 1).item()
                action_id = option_id + self.num_primitive_actions
                is_option = True
            
            return action_id, is_option, {
                'action_type_probs': action_type_probs,
                'primitive_logits': outputs['primitive_logits'],
                'option_logits': outputs['option_logits'],
                'combined_logits': outputs['combined_logits'],
            }


class OptionManager:
    """Manages the collection of available options and their execution."""
    
    def __init__(
        self,
        max_options: int = 8,
        min_success_rate: float = 0.15,
        evaluation_period: int = 50,  # Execute each option this many times before evaluation
    ):
        self.max_options = max_options
        self.min_success_rate = min_success_rate
        self.evaluation_period = evaluation_period
        
        # Option storage
        self.options: Dict[str, OptionSpec] = {}
        self.option_id_to_index: Dict[str, int] = {}
        self.index_to_option_id: Dict[int, str] = {}
        
        # Execution tracking
        self.current_execution: Optional[OptionExecution] = None
        self.execution_history: List[OptionExecution] = []
        
        # Statistics
        self.total_options_created = 0
        self.total_options_removed = 0
        self.option_evaluation_stats = {}
        
    def add_option(self, option_spec: OptionSpec) -> bool:
        """Add new option to the manager. Returns True if added successfully."""
        
        # Check capacity
        if len(self.options) >= self.max_options:
            # Try to remove worst performing option
            if not self._remove_worst_option():
                return False  # Couldn't make space
        
        # Assign index
        option_index = len(self.options)
        
        # Store option
        self.options[option_spec.option_id] = option_spec
        self.option_id_to_index[option_spec.option_id] = option_index
        self.index_to_option_id[option_index] = option_spec.option_id
        
        self.total_options_created += 1
        
        print(f"Added option '{option_spec.name}' (ID: {option_spec.option_id})")
        return True
    
    def remove_option(self, option_id: str) -> bool:
        """Remove option by ID."""
        if option_id not in self.options:
            return False
        
        # Remove from mappings
        option_index = self.option_id_to_index.pop(option_id)
        del self.index_to_option_id[option_index]
        del self.options[option_id]
        
        # Reindex remaining options
        self._reindex_options()
        
        self.total_options_removed += 1
        print(f"Removed option {option_id}")
        return True
    
    def get_available_options_mask(self, device: torch.device) -> torch.Tensor:
        """Get binary mask of available options."""
        mask = torch.zeros(self.max_options, device=device)
        for i in range(len(self.options)):
            if i in self.index_to_option_id:
                option_id = self.index_to_option_id[i]
                option = self.options[option_id]
                if option.is_viable():
                    mask[i] = 1.0
        return mask
    
    def start_option_execution(self, option_index: int, current_step: int) -> bool:
        """Start executing an option. Returns True if started successfully."""
        if self.current_execution is not None:
            return False  # Already executing an option
        
        if option_index not in self.index_to_option_id:
            return False  # Invalid option index
        
        option_id = self.index_to_option_id[option_index]
        option_spec = self.options[option_id]
        
        self.current_execution = OptionExecution(option_spec, current_step)
        return True
    
    def get_current_action(self) -> Optional[int]:
        """Get next primitive action from currently executing option."""
        if self.current_execution is None:
            return None
        
        return self.current_execution.get_next_action()
    
    def step_current_option(self, reward: float) -> bool:
        """Update current option execution. Returns True if option terminated."""
        if self.current_execution is None:
            return False
        
        should_terminate = self.current_execution.step(reward)
        
        if should_terminate:
            self._finish_current_option()
            return True
        
        return False
    
    def force_terminate_current_option(self, reason: str = "external_termination"):
        """Force termination of current option."""
        if self.current_execution is not None:
            self.current_execution.force_terminate(reason)
            self._finish_current_option()
    
    def _finish_current_option(self):
        """Complete current option execution and update statistics."""
        if self.current_execution is None:
            return
        
        execution = self.current_execution
        option_spec = execution.option_spec
        
        # Update option statistics
        option_spec.update_stats(
            duration=execution.steps_executed,
            reward=execution.accumulated_reward,
            success=execution.is_successful(),
        )
        
        # Store execution history
        self.execution_history.append(execution)
        if len(self.execution_history) > 200:
            self.execution_history.pop(0)
        
        self.current_execution = None
        
        print(f"Option '{option_spec.name}' finished: "
              f"{execution.termination_reason}, "
              f"reward={execution.accumulated_reward:.3f}, "
              f"steps={execution.steps_executed}")
    
    def _remove_worst_option(self) -> bool:
        """Remove the worst performing option. Returns True if removed."""
        if not self.options:
            return False
        
        # Find option with lowest success rate (among those with enough executions)
        candidate_options = [
            (option_id, option_spec) for option_id, option_spec in self.options.items()
            if option_spec.total_executions >= self.evaluation_period
        ]
        
        if not candidate_options:
            # No options have enough executions yet, remove random one
            option_id = next(iter(self.options.keys()))
        else:
            # Remove option with lowest success rate
            worst_option_id = min(candidate_options, key=lambda x: x[1].success_rate())[0]
            option_id = worst_option_id
        
        return self.remove_option(option_id)
    
    def _reindex_options(self):
        """Reindex options after removal."""
        new_id_to_index = {}
        new_index_to_id = {}
        
        for i, option_id in enumerate(self.options.keys()):
            new_id_to_index[option_id] = i
            new_index_to_id[i] = option_id
        
        self.option_id_to_index = new_id_to_index
        self.index_to_option_id = new_index_to_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stats = {
            'total_options': len(self.options),
            'max_options': self.max_options,
            'total_created': self.total_options_created,
            'total_removed': self.total_options_removed,
            'currently_executing': self.current_execution is not None,
        }
        
        if self.options:
            success_rates = [opt.success_rate() for opt in self.options.values()]
            stats.update({
                'avg_success_rate': np.mean(success_rates),
                'min_success_rate': np.min(success_rates),
                'max_success_rate': np.max(success_rates),
            })
        
        return stats
    
    def num_options(self) -> int:
        """Get current number of options."""
        return len(self.options)