"""Enhanced replay buffer with synthetic experience support for LoRe."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch


class SyntheticTransition:
    """Metadata for synthetic transitions following LoRe specification."""
    
    def __init__(
        self,
        is_synth: bool = False,
        w_synth: float = 1.0,
        advice_id: Optional[str] = None,
        origin: str = "env",
        llm_confidence: float = 0.0,
        execution_success: bool = True,
        synthetic_plan: Optional[Dict[str, Any]] = None,
    ):
        self.is_synth = is_synth
        self.w_synth = w_synth  # Importance weight for synthetic data
        self.advice_id = advice_id  # Unique ID for LLM advice
        self.origin = origin  # "env" | "llm" | "hybrid"
        self.llm_confidence = llm_confidence
        self.execution_success = execution_success
        self.synthetic_plan = synthetic_plan or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_synth": self.is_synth,
            "w_synth": self.w_synth,
            "advice_id": self.advice_id,
            "origin": self.origin,
            "llm_confidence": self.llm_confidence,
            "execution_success": self.execution_success,
            "synthetic_plan": self.synthetic_plan,
        }


class EnhancedReplayBuffer:
    """Enhanced replay buffer supporting synthetic transitions with LoRe specification."""
    
    def __init__(
        self,
        capacity: int,
        synthetic_ratio_max: float = 0.25,  # Max 25% synthetic data
        bc_regularization_coeff: float = 0.1,
        importance_sampling: bool = True,
        synthetic_weight_decay: float = 0.99,  # Decay synthetic weights over time
    ):
        self.capacity = capacity
        self.synthetic_ratio_max = synthetic_ratio_max
        self.bc_regularization_coeff = bc_regularization_coeff
        self.importance_sampling = importance_sampling
        self.synthetic_weight_decay = synthetic_weight_decay
        
        # Storage
        self.buffer: List[Dict[str, Any]] = []
        self.metadata: List[SyntheticTransition] = []
        self.position = 0
        self.size = 0
        
        # Synthetic data tracking
        self.synthetic_count = 0
        self.total_synthetic_added = 0
        self.synthetic_success_rate = 0.0
        
        # Importance sampling weights for synthetic data
        self.synthetic_weights_history: List[float] = []
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'synthetic_samples': 0,
            'synthetic_ratio': 0.0,
            'avg_synthetic_weight': 0.0,
            'synthetic_success_rate': 0.0,
        }
    
    def add_real(self, sample: Dict[str, Any]) -> None:
        """Add real environment transition."""
        metadata = SyntheticTransition(is_synth=False, origin="env")
        self._add_transition(sample, metadata)
    
    def add_synthetic(
        self,
        sample: Dict[str, Any],
        advice_id: str,
        llm_confidence: float = 0.5,
        execution_success: bool = True,
        synthetic_plan: Optional[Dict[str, Any]] = None,
        base_weight: float = 0.3,
    ) -> bool:
        """Add synthetic transition with LoRe controls.
        
        Returns:
            bool: True if added successfully, False if rejected due to ratio limits
        """
        current_ratio = self.synthetic_count / max(self.size, 1)
        
        # Reject if synthetic ratio too high
        if current_ratio >= self.synthetic_ratio_max:
            return False
        
        # Compute synthetic weight based on confidence and success
        confidence_factor = np.clip(llm_confidence, 0.1, 1.0)
        success_factor = 1.0 if execution_success else 0.5
        
        # Weight decays over time to prevent over-reliance
        time_factor = self.synthetic_weight_decay ** len(self.synthetic_weights_history)
        
        w_synth = base_weight * confidence_factor * success_factor * time_factor
        w_synth = np.clip(w_synth, 0.05, 0.8)  # Reasonable bounds
        
        metadata = SyntheticTransition(
            is_synth=True,
            w_synth=w_synth,
            advice_id=advice_id,
            origin="llm",
            llm_confidence=llm_confidence,
            execution_success=execution_success,
            synthetic_plan=synthetic_plan,
        )
        
        self._add_transition(sample, metadata)
        self.synthetic_count += 1
        self.total_synthetic_added += 1
        self.synthetic_weights_history.append(w_synth)
        
        # Update success rate
        if execution_success:
            self.synthetic_success_rate = (
                0.9 * self.synthetic_success_rate + 0.1 * 1.0
            )
        else:
            self.synthetic_success_rate = (
                0.9 * self.synthetic_success_rate + 0.1 * 0.0
            )
        
        return True
    
    def _add_transition(self, sample: Dict[str, Any], metadata: SyntheticTransition) -> None:
        """Internal method to add transition with metadata."""
        # Add synthetic metadata to sample
        for key, value in metadata.to_dict().items():
            sample[key] = torch.tensor(value) if isinstance(value, (int, float, bool)) else value
        
        if self.size < self.capacity:
            self.buffer.append(sample)
            self.metadata.append(metadata)
            self.size += 1
        else:
            # Replace oldest sample
            old_metadata = self.metadata[self.position]
            if old_metadata.is_synth:
                self.synthetic_count -= 1
            
            self.buffer[self.position] = sample
            self.metadata[self.position] = metadata
        
        self.position = (self.position + 1) % self.capacity
        self._update_stats()
    
    def sample(self, batch_size: int, prioritize_synthetic: bool = False) -> Dict[str, torch.Tensor]:
        """Sample batch with synthetic data controls."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        # Separate synthetic and real indices
        real_indices = [i for i, meta in enumerate(self.metadata[:self.size]) if not meta.is_synth]
        synthetic_indices = [i for i, meta in enumerate(self.metadata[:self.size]) if meta.is_synth]
        
        # Determine sampling composition
        max_synthetic = min(
            len(synthetic_indices),
            int(batch_size * self.synthetic_ratio_max)
        )
        
        if prioritize_synthetic and synthetic_indices:
            # Higher synthetic ratio when explicitly requested
            num_synthetic = min(max_synthetic, batch_size // 3)
        else:
            # Normal ratio-controlled sampling
            target_ratio = min(len(synthetic_indices) / max(self.size, 1), self.synthetic_ratio_max)
            num_synthetic = int(batch_size * target_ratio)
        
        num_real = batch_size - num_synthetic
        
        # Sample indices
        sampled_indices = []
        weights = []
        
        # Sample real data
        if num_real > 0 and real_indices:
            real_sample_indices = random.choices(real_indices, k=min(num_real, len(real_indices)))
            sampled_indices.extend(real_sample_indices)
            weights.extend([1.0] * len(real_sample_indices))
        
        # Sample synthetic data with importance weighting
        if num_synthetic > 0 and synthetic_indices:
            if self.importance_sampling:
                # Weight by synthetic importance and success
                synthetic_weights = []
                for idx in synthetic_indices:
                    meta = self.metadata[idx]
                    weight = meta.w_synth
                    if meta.execution_success:
                        weight *= 1.2  # Boost successful synthetic transitions
                    synthetic_weights.append(weight)
                
                # Normalize weights
                total_weight = sum(synthetic_weights)
                if total_weight > 0:
                    synthetic_weights = [w / total_weight for w in synthetic_weights]
                    synthetic_sample_indices = np.random.choice(
                        synthetic_indices,
                        size=min(num_synthetic, len(synthetic_indices)),
                        replace=True,
                        p=synthetic_weights
                    ).tolist()
                else:
                    synthetic_sample_indices = random.choices(synthetic_indices, k=min(num_synthetic, len(synthetic_indices)))
            else:
                synthetic_sample_indices = random.choices(synthetic_indices, k=min(num_synthetic, len(synthetic_indices)))
            
            sampled_indices.extend(synthetic_sample_indices)
            # Add synthetic weights
            for idx in synthetic_sample_indices:
                weights.append(self.metadata[idx].w_synth)
        
        # Pad if necessary
        while len(sampled_indices) < batch_size and real_indices:
            idx = random.choice(real_indices)
            sampled_indices.append(idx)
            weights.append(1.0)
        
        # Collect samples
        samples = [self.buffer[idx] for idx in sampled_indices]
        
        # Convert to TensorDict format
        try:
            from tensordict import TensorDict
        except ImportError:
            from torchrl.data import TensorDict
        
        # Batch samples
        batch_data = {}
        for key in samples[0].keys():
            if key in ['observation', 'action', 'reward', 'done'] or key.startswith('llm_'):
                values = [sample[key] for sample in samples]
                if isinstance(values[0], torch.Tensor):
                    batch_data[key] = torch.stack(values)
                else:
                    batch_data[key] = torch.tensor(values)
        
        # Add sampling metadata
        batch_data['_sample_weights'] = torch.tensor(weights, dtype=torch.float32)
        batch_data['_sample_indices'] = torch.tensor(sampled_indices, dtype=torch.long)
        batch_data['_is_synthetic'] = torch.tensor(
            [self.metadata[idx].is_synth for idx in sampled_indices], 
            dtype=torch.bool
        )
        
        return TensorDict(batch_data, batch_size=[batch_size])
    
    def get_bc_regularization_batch(self, batch_size: int = 32) -> Optional[Dict[str, torch.Tensor]]:
        """Get batch of synthetic data for behavioral cloning regularization."""
        synthetic_indices = [i for i, meta in enumerate(self.metadata[:self.size]) if meta.is_synth]
        
        if len(synthetic_indices) < batch_size // 2:
            return None
        
        # Sample synthetic transitions
        sample_indices = random.choices(synthetic_indices, k=min(batch_size, len(synthetic_indices)))
        samples = [self.buffer[idx] for idx in sample_indices]
        
        try:
            from tensordict import TensorDict
        except ImportError:
            from torchrl.data import TensorDict
        
        # Batch samples for BC loss
        batch_data = {}
        for key in ['observation', 'action']:
            if key in samples[0]:
                values = [sample[key] for sample in samples]
                if isinstance(values[0], torch.Tensor):
                    batch_data[key] = torch.stack(values)
                else:
                    batch_data[key] = torch.tensor(values)
        
        # Add LLM logits for BC target if available
        if 'llm_prior_logits' in samples[0]:
            values = [sample['llm_prior_logits'] for sample in samples]
            batch_data['llm_prior_logits'] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
        
        return TensorDict(batch_data, batch_size=[len(samples)]) if batch_data else None
    
    def _update_stats(self) -> None:
        """Update buffer statistics."""
        self.stats['total_samples'] = self.size
        self.stats['synthetic_samples'] = self.synthetic_count
        self.stats['synthetic_ratio'] = self.synthetic_count / max(self.size, 1)
        
        if self.synthetic_weights_history:
            self.stats['avg_synthetic_weight'] = np.mean(self.synthetic_weights_history[-100:])
        else:
            self.stats['avg_synthetic_weight'] = 0.0
        
        self.stats['synthetic_success_rate'] = self.synthetic_success_rate
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics for monitoring."""
        return self.stats.copy()
    
    def clear_old_synthetic(self, max_age: int = 10000) -> int:
        """Clear old synthetic data to prevent staleness. Returns number removed."""
        if len(self.synthetic_weights_history) <= max_age:
            return 0
        
        removed = 0
        # Simple strategy: remove oldest 10% of synthetic data
        target_remove = max(1, self.synthetic_count // 10)
        
        new_buffer = []
        new_metadata = []
        
        for i, (sample, meta) in enumerate(zip(self.buffer[:self.size], self.metadata[:self.size])):
            if meta.is_synth and removed < target_remove:
                removed += 1
                self.synthetic_count -= 1
            else:
                new_buffer.append(sample)
                new_metadata.append(meta)
        
        self.buffer = new_buffer + self.buffer[self.size:]
        self.metadata = new_metadata + self.metadata[self.size:]
        self.size = len(new_buffer)
        self.position = self.size % self.capacity
        
        # Update weights history
        self.synthetic_weights_history = self.synthetic_weights_history[-max_age//2:]
        
        self._update_stats()
        return removed
    
    def __len__(self) -> int:
        return self.size