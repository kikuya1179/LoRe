from __future__ import annotations

from typing import Optional


def make_replay(capacity: int) -> "torchrl.data.ReplayBuffer":  # type: ignore[name-defined]
    try:
        from torchrl.data import ReplayBuffer
        from torchrl.data.replay_buffers.storages import LazyMemmapStorage
    except Exception as e:  # pragma: no cover
        raise RuntimeError("TorchRL is required for ReplayBuffer") from e

    storage = LazyMemmapStorage(capacity)
    rb = ReplayBuffer(storage=storage)
    return rb

