"""Reward shaping and HER (Hindsight Experience Replay) support for LLM-enhanced RL."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .code_out import RBExtra


@dataclass
class SubgoalSpec:
    """Subgoal specification for HER."""
    id: str
    description: str
    check_fn: Optional[callable] = None  # Function to check achievement
    reward_fn: Optional[callable] = None  # Function to compute reward
    max_steps: int = 100


class RewardShaper:
    """Potential-based reward shaping to maintain optimal policy invariance."""
    
    def __init__(self, beta: float = 0.1, clip_range: float = 0.2):
        self.beta = beta
        self.clip_range = clip_range
        self.potential_cache: Dict[str, float] = {}
    
    def compute_shaped_reward(self, env_reward: float, r_shaped: float, 
                            state_potential: float = 0.0, 
                            next_state_potential: float = 0.0,
                            gamma: float = 0.99) -> float:
        """Compute total reward with potential-based shaping."""
        # Clip LLM's shaped reward to prevent exploitation
        r_shaped_clipped = np.clip(r_shaped, -self.clip_range, self.clip_range)
        
        # Potential-based shaping: F(s') - gamma * F(s)
        potential_diff = next_state_potential - gamma * state_potential
        
        # Total reward
        total_reward = env_reward + self.beta * (r_shaped_clipped + potential_diff)
        
        return float(total_reward)
    
    def compute_potential(self, obs: np.ndarray, context: Dict[str, Any]) -> float:
        """Compute state potential function."""
        # Simple potential based on game progress
        # This should be domain-specific for Crafter
        
        # Health component
        health_ratio = context.get('health', 100) / 100.0
        health_potential = health_ratio * 0.1
        
        # Inventory component (more items = higher potential)
        inventory = context.get('inventory', {})
        inventory_potential = min(0.5, len(inventory) * 0.05)
        
        # Goal progress component
        goals = context.get('goals', {})
        goal_potential = sum(goals.values()) * 0.1
        
        total_potential = health_potential + inventory_potential + goal_potential
        return float(np.clip(total_potential, 0.0, 1.0))


class HERProcessor:
    """Hindsight Experience Replay processor for goal-conditioned learning."""
    
    def __init__(self, her_ratio: float = 0.8, max_subgoals: int = 10):
        self.her_ratio = her_ratio
        self.max_subgoals = max_subgoals
        self.subgoal_registry: Dict[str, SubgoalSpec] = {}
        self._init_crafter_subgoals()
    
    def _init_crafter_subgoals(self) -> None:
        """Initialize Crafter-specific subgoals."""
        crafter_subgoals = [
            SubgoalSpec(
                id="collect_wood",
                description="Collect wood from trees",
                check_fn=lambda ctx: ctx.get('inventory', {}).get('wood', 0) > 0,
                reward_fn=lambda ctx: 0.1 if ctx.get('inventory', {}).get('wood', 0) > 0 else 0.0,
                max_steps=50
            ),
            SubgoalSpec(
                id="collect_stone", 
                description="Collect stone from rocks",
                check_fn=lambda ctx: ctx.get('inventory', {}).get('stone', 0) > 0,
                reward_fn=lambda ctx: 0.15 if ctx.get('inventory', {}).get('stone', 0) > 0 else 0.0,
                max_steps=75
            ),
            SubgoalSpec(
                id="make_workbench",
                description="Craft a workbench",
                check_fn=lambda ctx: ctx.get('inventory', {}).get('workbench', 0) > 0,
                reward_fn=lambda ctx: 0.3 if ctx.get('inventory', {}).get('workbench', 0) > 0 else 0.0,
                max_steps=100
            ),
            SubgoalSpec(
                id="make_tool",
                description="Craft any tool",
                check_fn=lambda ctx: any(
                    tool in ctx.get('inventory', {}) 
                    for tool in ['stone_pickaxe', 'stone_axe', 'wood_pickaxe', 'wood_axe']
                ),
                reward_fn=lambda ctx: 0.25,
                max_steps=80
            ),
            SubgoalSpec(
                id="find_water",
                description="Locate water source", 
                check_fn=lambda ctx: ctx.get('near_water', False),
                reward_fn=lambda ctx: 0.1,
                max_steps=60
            ),
        ]
        
        for subgoal in crafter_subgoals:
            self.subgoal_registry[subgoal.id] = subgoal
    
    def register_subgoal(self, subgoal: SubgoalSpec) -> None:
        """Register a new subgoal."""
        self.subgoal_registry[subgoal.id] = subgoal
    
    def generate_her_transitions(self, episode_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate HER transitions from episode data."""
        if len(episode_data) < 2:
            return []
        
        her_transitions = []
        
        # Extract achieved subgoals during episode
        achieved_subgoals = self._extract_achieved_subgoals(episode_data)
        
        if not achieved_subgoals:
            return []
        
        # Generate HER transitions
        for transition in episode_data:
            if random.random() > self.her_ratio:
                continue
            
            # Pick random achieved subgoal as retrospective goal
            target_subgoal = random.choice(achieved_subgoals)
            subgoal_spec = self.subgoal_registry.get(target_subgoal)\n            \n            if subgoal_spec and subgoal_spec.reward_fn:\n                # Create modified transition with hindsight reward\n                her_transition = transition.copy()\n                \n                # Update reward based on retrospective goal achievement\n                context = transition.get('context', {})\n                hindsight_reward = subgoal_spec.reward_fn(context)\n                \n                # Mark as HER transition\n                her_transition['her_subgoal'] = target_subgoal\n                her_transition['her_reward'] = hindsight_reward\n                her_transition['is_her'] = True\n                \n                her_transitions.append(her_transition)\n        \n        return her_transitions\n    \n    def _extract_achieved_subgoals(self, episode_data: List[Dict[str, Any]]) -> List[str]:\n        \"\"\"Extract subgoals that were achieved during the episode.\"\"\"\n        achieved = []\n        \n        for transition in episode_data:\n            context = transition.get('context', {})\n            \n            for subgoal_id, subgoal_spec in self.subgoal_registry.items():\n                if subgoal_spec.check_fn and subgoal_spec.check_fn(context):\n                    if subgoal_id not in achieved:\n                        achieved.append(subgoal_id)\n        \n        return achieved\n    \n    def compute_subgoal_reward(self, subgoal_id: str, context: Dict[str, Any], \n                              step_count: int) -> float:\n        \"\"\"Compute reward for specific subgoal achievement.\"\"\"\n        subgoal_spec = self.subgoal_registry.get(subgoal_id)\n        \n        if not subgoal_spec:\n            return 0.0\n        \n        # Check if achieved\n        if subgoal_spec.check_fn and subgoal_spec.check_fn(context):\n            base_reward = subgoal_spec.reward_fn(context) if subgoal_spec.reward_fn else 0.1\n            \n            # Time bonus (faster achievement = higher reward)\n            time_bonus = max(0.0, 1.0 - step_count / subgoal_spec.max_steps) * 0.05\n            \n            return base_reward + time_bonus\n        \n        # Partial reward for progress\n        progress_reward = self._compute_progress_reward(subgoal_id, context, step_count)\n        return progress_reward\n    \n    def _compute_progress_reward(self, subgoal_id: str, context: Dict[str, Any], \n                               step_count: int) -> float:\n        \"\"\"Compute partial reward for progress toward subgoal.\"\"\"\n        # Domain-specific progress computation\n        if subgoal_id == \"collect_wood\":\n            # Reward for being near trees\n            if context.get('near_trees', False):\n                return 0.01\n        elif subgoal_id == \"collect_stone\":\n            # Reward for being near rocks\n            if context.get('near_rocks', False):\n                return 0.01\n        elif subgoal_id == \"make_workbench\":\n            # Reward for having wood (prerequisite)\n            if context.get('inventory', {}).get('wood', 0) > 0:\n                return 0.02\n        elif subgoal_id == \"find_water\":\n            # Reward for exploration (simplified)\n            exploration_bonus = min(0.005, step_count * 0.0001)\n            return exploration_bonus\n        \n        return 0.0\n\n\nclass EnhancedReplayProcessor:\n    \"\"\"Enhanced replay buffer processor with reward shaping and HER.\"\"\"\n    \n    def __init__(self, reward_shaper: Optional[RewardShaper] = None,\n                 her_processor: Optional[HERProcessor] = None):\n        self.reward_shaper = reward_shaper or RewardShaper()\n        self.her_processor = her_processor or HERProcessor()\n        self.episode_buffer: List[Dict[str, Any]] = []\n    \n    def process_transition(self, transition_data: Dict[str, Any], \n                          rb_extra: Optional[RBExtra] = None) -> Dict[str, Any]:\n        \"\"\"Process single transition with reward shaping and subgoal tracking.\"\"\"\n        processed = transition_data.copy()\n        \n        if rb_extra:\n            # Apply reward shaping\n            env_reward = float(processed.get('reward', 0.0))\n            shaped_reward = self.reward_shaper.compute_shaped_reward(\n                env_reward=env_reward,\n                r_shaped=rb_extra.r_shaped,\n                state_potential=0.0,  # TODO: Implement state potential\n                next_state_potential=0.0,\n            )\n            \n            processed['total_reward'] = shaped_reward\n            processed['env_reward'] = env_reward\n            processed['shaped_component'] = shaped_reward - env_reward\n            \n            # Add subgoal tracking\n            if rb_extra.subgoal_id >= 0:\n                subgoal_ids = list(self.her_processor.subgoal_registry.keys())\n                if rb_extra.subgoal_id < len(subgoal_ids):\n                    subgoal_name = subgoal_ids[rb_extra.subgoal_id]\n                    processed['active_subgoal'] = subgoal_name\n                    processed['subgoal_steps'] = rb_extra.subgoal_tau\n        \n        return processed\n    \n    def add_to_episode(self, transition_data: Dict[str, Any]) -> None:\n        \"\"\"Add transition to episode buffer.\"\"\"\n        self.episode_buffer.append(transition_data)\n    \n    def finalize_episode(self) -> List[Dict[str, Any]]:\n        \"\"\"Finalize episode and generate HER transitions.\"\"\"\n        if not self.episode_buffer:\n            return []\n        \n        # Original transitions\n        all_transitions = self.episode_buffer.copy()\n        \n        # Generate HER transitions\n        her_transitions = self.her_processor.generate_her_transitions(self.episode_buffer)\n        all_transitions.extend(her_transitions)\n        \n        # Clear episode buffer\n        self.episode_buffer.clear()\n        \n        return all_transitions\n    \n    def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get processing statistics.\"\"\"\n        return {\n            'episode_length': len(self.episode_buffer),\n            'registered_subgoals': len(self.her_processor.subgoal_registry),\n            'reward_shaping_beta': self.reward_shaper.beta,\n            'her_ratio': self.her_processor.her_ratio,\n        }