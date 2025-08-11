"""GPU-optimized replay buffer implementation."""

from __future__ import annotations

import torch
from typing import Optional, Dict, Any
from tensordict import TensorDict


class GPUReplayBuffer:
    """GPU-optimized circular replay buffer with minimal CPU/GPU transfers."""
    
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors on GPU
        self._storage: Dict[str, torch.Tensor] = {}
        self._initialized = False
    
    def _initialize_storage(self, sample: TensorDict) -> None:
        """Initialize GPU storage based on first sample."""
        if self._initialized:
            return
            
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                # Pre-allocate tensor on GPU
                shape = (self.capacity,) + value.shape
                self._storage[key] = torch.zeros(shape, dtype=value.dtype, device=self.device)
            elif isinstance(value, dict) and key == "next":
                # Handle nested structure
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        shape = (self.capacity,) + sub_value.shape
                        full_key = f"next_{sub_key}"
                        self._storage[full_key] = torch.zeros(shape, dtype=sub_value.dtype, device=self.device)
        
        self._initialized = True
    
    def add(self, sample: TensorDict) -> None:
        """Add sample to buffer (GPU operations only)."""
        # Ensure sample is on correct device
        sample = sample.to(self.device, non_blocking=True)
        
        # Initialize storage if needed
        if not self._initialized:
            self._initialize_storage(sample)
        
        # Store data
        for key, value in sample.items():
            if key in self._storage:
                self._storage[key][self.position] = value
            elif isinstance(value, dict) and key == "next":
                for sub_key, sub_value in value.items():
                    full_key = f"next_{sub_key}"
                    if full_key in self._storage:
                        self._storage[full_key][self.position] = sub_value
        
        # Update pointers
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> TensorDict:
        """Sample batch entirely on GPU."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        # Random sampling on GPU
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # Gather samples
        batch_data = {}
        for key, storage in self._storage.items():
            batch_data[key] = storage[indices]
        
        # Reconstruct nested structure
        result_data = {}
        next_data = {}
        for key, value in batch_data.items():
            if key.startswith("next_"):
                sub_key = key[5:]  # Remove "next_" prefix
                next_data[sub_key] = value
            else:
                result_data[key] = value
        
        if next_data:
            result_data["next"] = next_data
        
        return TensorDict(result_data, batch_size=[batch_size])
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self) -> None:
        """Clear buffer."""
        self.position = 0
        self.size = 0
    
    def memory_usage(self) -> int:
        """Estimate GPU memory usage in bytes."""
        total_bytes = 0
        for storage in self._storage.values():
            total_bytes += storage.nelement() * storage.element_size()
        return total_bytes


class HybridReplayBuffer:
    """Hybrid replay buffer that uses GPU storage but falls back to CPU if needed."""
    
    def __init__(self, capacity: int, device: torch.device, gpu_capacity_ratio: float = 0.8):
        self.capacity = capacity
        self.device = device
        self.gpu_capacity = int(capacity * gpu_capacity_ratio)
        
        # Try GPU buffer first
        try:
            self.gpu_buffer = GPUReplayBuffer(self.gpu_capacity, device)
            self.use_gpu = True
        except RuntimeError:
            self.use_gpu = False
        
        # Fallback storage
        self._cpu_storage = []
        self._cpu_position = 0
        self._cpu_size = 0
    
    def add(self, sample: TensorDict) -> None:
        """Add sample with GPU preference."""
        if self.use_gpu:
            try:
                self.gpu_buffer.add(sample)
                return
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Fall back to CPU storage
                self.use_gpu = False
        
        # CPU fallback
        sample_cpu = sample.cpu()
        if self._cpu_size < self.capacity:
            self._cpu_storage.append(sample_cpu)
            self._cpu_size += 1
        else:
            self._cpu_storage[self._cpu_position] = sample_cpu
        
        self._cpu_position = (self._cpu_position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> TensorDict:
        """Sample from appropriate storage."""
        if self.use_gpu and len(self.gpu_buffer) > 0:
            return self.gpu_buffer.sample(batch_size)
        
        # CPU sampling
        if self._cpu_size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        indices = torch.randint(0, self._cpu_size, (batch_size,))
        samples = [self._cpu_storage[idx] for idx in indices]
        
        # Stack samples
        batch_data = {}
        for key in samples[0].keys():
            if key == "next":
                next_data = {}
                for sub_key in samples[0]["next"].keys():
                    values = [s["next"][sub_key] for s in samples]
                    next_data[sub_key] = torch.stack(values)
                batch_data["next"] = next_data
            else:
                values = [s[key] for s in samples]
                batch_data[key] = torch.stack(values)
        
        return TensorDict(batch_data, batch_size=[batch_size])
    
    def __len__(self) -> int:
        if self.use_gpu:
            return len(self.gpu_buffer)
        return self._cpu_size


def create_optimized_replay(capacity: int, device: torch.device, force_gpu: bool = False) -> Any:
    """Create the most appropriate replay buffer for the system."""
    if force_gpu:
        return GPUReplayBuffer(capacity, device)
    else:
        return HybridReplayBuffer(capacity, device)