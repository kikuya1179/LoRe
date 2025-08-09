from __future__ import annotations

import os
from typing import Optional


def make_replay(capacity: int) -> "torchrl.data.ReplayBuffer":  # type: ignore[name-defined]
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

