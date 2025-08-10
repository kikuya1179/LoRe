"""CodeOut schema and RBExtra structures for LLM-enhanced RL."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pathlib import Path


@dataclass
class CodeOut:
    """LLM code execution output (JSON-like structure)."""
    
    features: List[float] = field(default_factory=list)  # f1, f2, ..., fK
    subgoal: Optional[Dict[str, Any]] = None  # {"id": "cook_wood", "tau": 8}
    r_shaped: float = 0.0  # Potential-based reward shaping [-0.2, 0.2]
    policy: Optional[Dict[str, Any]] = None  # {"logits": [...], "temp": 1.0}
    mask: Optional[List[int]] = None  # [0,1,1,1,0] action mask
    confidence: float = 0.0  # [0.0, 1.0]
    notes: str = ""  # Human-readable explanation
    
    def validate(self, num_actions: int, max_features: int = 32) -> bool:
        """Validate schema constraints."""
        try:
            # Features: normalize to [-3, 3] range
            if len(self.features) > max_features:
                return False
            for f in self.features:
                if not isinstance(f, (int, float)) or abs(f) > 3.0:
                    return False
            
            # r_shaped bounds
            if not isinstance(self.r_shaped, (int, float)) or abs(self.r_shaped) > 0.2:
                return False
            
            # Policy logits size
            if self.policy is not None:
                logits = self.policy.get("logits", [])
                if len(logits) != num_actions:
                    return False
            
            # Mask size
            if self.mask is not None:
                if len(self.mask) != num_actions:
                    return False
                if not all(m in [0, 1] for m in self.mask):
                    return False
            
            # Confidence bounds
            if not 0.0 <= self.confidence <= 1.0:
                return False
            
            return True
        except Exception:
            return False


@dataclass
class RBExtra:
    """Lightweight LLM data for replay buffer (memory-efficient)."""
    
    code_id: int = 0  # External KV reference
    conf: float = 0.0  # float16 equivalent
    r_shaped: float = 0.0
    subgoal_id: int = -1  # -1 means no subgoal
    subgoal_tau: int = 0
    feat: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    logits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    mask_bits: int = 0  # Bitmask for up to 64 actions
    
    @classmethod
    def from_code_out(cls, code_out: CodeOut, code_id: int, num_actions: int) -> 'RBExtra':
        """Convert CodeOut to memory-efficient RBExtra."""
        # Features
        feat = np.array(code_out.features, dtype=np.float32)
        
        # Logits
        logits = np.array([], dtype=np.float32)
        if code_out.policy and "logits" in code_out.policy:
            logits = np.array(code_out.policy["logits"], dtype=np.float32)
        
        # Mask to bitmask (up to 64 actions)
        mask_bits = 0
        if code_out.mask and num_actions <= 64:
            for i, m in enumerate(code_out.mask):
                if m:
                    mask_bits |= (1 << i)
        
        # Subgoal
        subgoal_id = -1
        subgoal_tau = 0
        if code_out.subgoal:
            # Simple hash-based ID (could use proper registry)
            subgoal_str = str(code_out.subgoal.get("id", ""))
            subgoal_id = hash(subgoal_str) % (2**15)  # Keep it int16 range
            subgoal_tau = min(255, int(code_out.subgoal.get("tau", 0)))
        
        return cls(
            code_id=code_id,
            conf=float(code_out.confidence),
            r_shaped=float(code_out.r_shaped),
            subgoal_id=subgoal_id,
            subgoal_tau=subgoal_tau,
            feat=feat,
            logits=logits,
            mask_bits=mask_bits,
        )
    
    def get_mask(self, num_actions: int) -> Optional[np.ndarray]:
        """Convert bitmask back to action mask array."""
        if self.mask_bits == 0 or num_actions > 64:
            return None
        
        mask = np.zeros(num_actions, dtype=np.int32)
        for i in range(num_actions):
            if self.mask_bits & (1 << i):
                mask[i] = 1
        return mask


class CodeKVStore:
    """External key-value store for LLM code and metadata."""
    
    def __init__(self, store_path: str = "code_cache"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        self._id_counter = 0
        self._cache: Dict[int, str] = {}
    
    def store_code(self, code: str) -> int:
        """Store code and return unique ID."""
        # Use hash as filename for deduplication
        code_hash = hashlib.md5(code.encode()).hexdigest()
        file_path = self.store_path / f"{code_hash}.txt"
        
        # Check if already exists
        if file_path.exists():
            # Read existing ID from metadata
            meta_path = self.store_path / f"{code_hash}.meta"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        existing_id = int(f.read().strip())
                    self._cache[existing_id] = code
                    return existing_id
                except Exception:
                    pass
        
        # Store new code
        self._id_counter += 1
        code_id = self._id_counter
        
        # Write code file
        with open(file_path, 'w') as f:
            f.write(code)
        
        # Write metadata
        meta_path = self.store_path / f"{code_hash}.meta"
        with open(meta_path, 'w') as f:
            f.write(str(code_id))
        
        self._cache[code_id] = code
        return code_id
    
    def get_code(self, code_id: int) -> Optional[str]:
        """Retrieve code by ID."""
        if code_id in self._cache:
            return self._cache[code_id]
        
        # Search filesystem (slower fallback)
        for meta_file in self.store_path.glob("*.meta"):
            try:
                with open(meta_file, 'r') as f:
                    stored_id = int(f.read().strip())
                if stored_id == code_id:
                    code_hash = meta_file.stem
                    code_file = self.store_path / f"{code_hash}.txt"
                    if code_file.exists():
                        with open(code_file, 'r') as f:
                            code = f.read()
                        self._cache[code_id] = code
                        return code
            except Exception:
                continue
        
        return None
    
    def clear_cache(self) -> None:
        """Clear memory cache (filesystem remains)."""
        self._cache.clear()


def compute_state_hash(obs: np.ndarray, stats: Optional[Dict[str, Any]] = None) -> str:
    """Compute state hash for caching LLM outputs."""
    # Simple hash based on observation
    obs_bytes = obs.tobytes() if hasattr(obs, 'tobytes') else str(obs).encode()
    hasher = hashlib.md5(obs_bytes)
    
    if stats:
        stats_str = str(sorted(stats.items()))
        hasher.update(stats_str.encode())
    
    return hasher.hexdigest()