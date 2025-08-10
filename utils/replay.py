from __future__ import annotations

import os
import random
from typing import Optional, List, Dict, Any
import numpy as np
import torch


class PriorityReplayBuffer:
    """Priority Experience Replay buffer with TD-error based sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent  
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small constant to avoid zero priorities
        
        # Storage
        self.buffer: List[Dict[str, Any]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # For efficient sampling
        self._max_priority = 1.0
    
    def add(self, sample: Dict[str, Any], td_error: float = None) -> None:
        """Add sample with optional TD error for priority."""
        # Compute priority
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = self._max_priority  # New samples get max priority
        
        # Store
        if self.size < self.capacity:
            self.buffer.append(sample)
            self.size += 1
        else:
            self.buffer[self.position] = sample
        
        self.priorities[self.position] = priority
        self._max_priority = max(self._max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with priority-based selection."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=True)
        
        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Collect samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Convert to TensorDict format
        try:
            from tensordict import TensorDict
        except ImportError:
            from torchrl.data import TensorDict
        
        # Batch samples
        batch_data = {}
        for key in samples[0].keys():
            if key.startswith('llm_') or key in ['observation', 'action', 'reward', 'done']:
                values = [sample[key] for sample in samples]
                if isinstance(values[0], torch.Tensor):
                    batch_data[key] = torch.stack(values)
                else:
                    batch_data[key] = torch.tensor(values)
        
        # Add importance weights and indices for priority updates
        batch_data['_weights'] = torch.tensor(weights, dtype=torch.float32)
        batch_data['_indices'] = torch.tensor(indices, dtype=torch.long)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return TensorDict(batch_data, batch_size=[batch_size])
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """Update priorities based on new TD errors."""
        indices = indices.cpu().numpy()
        td_errors = td_errors.detach().cpu().numpy()
        
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < self.size:
                priority = (abs(td_error) + self.epsilon) ** self.alpha
                self.priorities[idx] = priority
                self._max_priority = max(self._max_priority, priority)
    
    def __len__(self) -> int:
        return self.size


class NoveltyTracker:
    """Track state novelty for triggering LLM calls."""
    
    def __init__(self, history_size: int = 1000, novelty_threshold: float = 0.1):
        self.history_size = history_size
        self.novelty_threshold = novelty_threshold
        self.state_history: List[np.ndarray] = []
        self.feature_stats = {'mean': 0.0, 'std': 1.0}
    
    def compute_novelty(self, state_features: np.ndarray) -> float:
        """Compute novelty score for state."""
        if len(self.state_history) == 0:
            return 1.0  # First state is always novel
        
        # Compute distances to historical states  
        distances = []
        for hist_state in self.state_history[-100:]:  # Check last 100 states
            if hist_state.shape == state_features.shape:
                dist = np.linalg.norm(state_features - hist_state)
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Novelty is minimum distance to any historical state
        min_distance = min(distances)
        
        # Normalize by running statistics
        novelty = min_distance / max(self.feature_stats['std'], 0.1)
        return float(np.clip(novelty, 0.0, 1.0))
    
    def add_state(self, state_features: np.ndarray) -> None:
        """Add state to history."""
        self.state_history.append(state_features.copy())
        
        # Maintain history size
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)
        
        # Update running statistics
        if len(self.state_history) > 10:
            all_states = np.array(self.state_history)
            self.feature_stats['mean'] = float(np.mean(all_states))
            self.feature_stats['std'] = float(np.std(all_states))
    
    def is_novel(self, state_features: np.ndarray) -> bool:
        """Check if state is novel enough to trigger LLM."""
        novelty = self.compute_novelty(state_features)
        return novelty > self.novelty_threshold


class TDErrorTracker:
    """Track TD errors for triggering LLM calls."""
    
    def __init__(self, error_history_size: int = 1000, error_threshold: float = 0.5):
        self.error_history_size = error_history_size
        self.error_threshold = error_threshold
        self.error_history: List[float] = []
        self.running_mean = 0.0
        self.running_std = 1.0
    
    def add_error(self, td_error: float) -> None:
        """Add TD error to history."""
        self.error_history.append(float(td_error))
        
        # Maintain history size
        if len(self.error_history) > self.error_history_size:
            self.error_history.pop(0)
        
        # Update running statistics
        if len(self.error_history) > 10:
            self.running_mean = float(np.mean(self.error_history))
            self.running_std = float(np.std(self.error_history) or 1.0)
    
    def is_high_error(self, td_error: float) -> bool:
        """Check if TD error is high enough to trigger LLM."""
        if len(self.error_history) < 10:
            return True  # Bootstrap period
        
        # Standardized error
        z_score = (td_error - self.running_mean) / self.running_std
        return z_score > self.error_threshold
    
    def get_percentile_threshold(self, percentile: float = 90.0) -> float:
        """Get threshold for top percentile of errors."""
        if len(self.error_history) < 10:
            return 0.0
        
        return float(np.percentile(self.error_history, percentile))


def make_replay(capacity: int, use_priority: bool = False) -> "torchrl.data.ReplayBuffer":  # type: ignore[name-defined]
    try:
        from torchrl.data import ReplayBuffer
        from torchrl.data.replay_buffers.storages import (
            LazyMemmapStorage,
            LazyTensorStorage,
        )
        # ListStorage はバージョンにより存在しないことがあるため任意
        try:
            from torchrl.data.replay_buffers.storages import ListStorage  # type: ignore
        except Exception:  # pragma: no cover
            ListStorage = None  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("TorchRL is required for ReplayBuffer") from e

    if use_priority:
        # Return our custom priority buffer wrapped
        priority_rb = PriorityReplayBuffer(capacity)
        return priority_rb  # type: ignore
    
    backend = os.environ.get("LORE_REPLAY_BACKEND", "auto").lower()

    if backend == "list" and 'ListStorage' in locals() and ListStorage is not None:
        storage = ListStorage(capacity)  # type: ignore[misc]
    elif backend == "tensor" or (backend == "auto" and os.name == "nt"):
        # Windows ではデフォルトをメモリ（テンソル）にして memmap の WinError 1455 を回避
        storage = LazyTensorStorage(capacity)
    else:
        storage = LazyMemmapStorage(capacity)

    rb = ReplayBuffer(storage=storage)
    return rb

