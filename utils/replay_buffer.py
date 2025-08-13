from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    done: bool
    next_obs: np.ndarray


class ReplayBuffer:
    """Simple episodic replay buffer with sequence sampling.

    Stores transitions in contiguous arrays and maintains episode indices
    to support sequence sampling without crossing episode boundaries.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int]) -> None:
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)

        self.idx = 0
        self.full = False

        # episode boundaries as indices of terminal transitions
        self.terminal_indices: List[int] = []
        # success tags for transitions
        self.success_flags = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool, next_obs: np.ndarray, *, success: bool = False) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = 1.0 if done else 0.0
        self.next_obs[i] = next_obs
        self.success_flags[i] = 1.0 if success else 0.0

        if done:
            self.terminal_indices.append(i)
            if len(self.terminal_indices) > 2 * (self.capacity // 10):  # cap list to reasonable size
                self.terminal_indices = self.terminal_indices[-(self.capacity // 10):]

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def _is_valid_start(self, start: int, seq_len: int) -> bool:
        end = start + seq_len - 1
        if end >= len(self):
            return False
        # invalid if any terminal inside [start, end-1]
        # allow end to coincide with terminal (sequence can end at done)
        for t in self.terminal_indices:
            if start <= t < end:
                return False
        return True

    def sample_sequences(self, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        assert len(self) >= batch_size * seq_len, "Not enough data in replay buffer"

        max_start = len(self) - seq_len
        starts: List[int] = []
        trials = 0
        # Success-biased sampling: pick a portion from near successful transitions
        # 1) gather candidate indices with success
        success_idxs = np.where(self.success_flags[:len(self)] > 0.5)[0]
        back_steps = 10
        if success_idxs.size > 0:
            # expand to windows before success
            candidates = []
            for si in success_idxs.tolist():
                start = max(0, si - back_steps)
                end = max(0, si - 1)
                for s in range(start, min(end, max_start) + 1):
                    if self._is_valid_start(s, seq_len):
                        candidates.append(s)
            random.shuffle(candidates)
            while len(starts) < batch_size // 2 and candidates:
                starts.append(candidates.pop())
        # 2) fill the rest uniformly
        while len(starts) < batch_size and trials < batch_size * 200:
            s = random.randint(0, max_start)
            if self._is_valid_start(s, seq_len):
                starts.append(s)
            trials += 1
        if len(starts) < batch_size:
            # fallback: allow crossing episodes if necessary
            starts = [random.randint(0, max_start) for _ in range(batch_size)]

        obs = np.stack([self.obs[s:s+seq_len] for s in starts], axis=0)  # [B,T,C,H,W]
        next_obs = np.stack([self.next_obs[s:s+seq_len] for s in starts], axis=0)
        actions = np.stack([self.actions[s:s+seq_len] for s in starts], axis=0)
        rewards = np.stack([self.rewards[s:s+seq_len] for s in starts], axis=0)
        dones = np.stack([self.dones[s:s+seq_len] for s in starts], axis=0)

        batch = {
            "observation_seq": torch.from_numpy(obs).to(device).float(),
            "next_observation_seq": torch.from_numpy(next_obs).to(device).float(),
            "action_seq": torch.from_numpy(actions).to(device).long(),
            "reward_seq": torch.from_numpy(rewards).to(device).float(),
            "done_seq": torch.from_numpy(dones).to(device).float(),
        }
        return batch


