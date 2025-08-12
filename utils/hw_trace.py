from __future__ import annotations

from typing import Any, Optional, Dict

_current_step: int = -1


def set_log_path(path: str) -> None:
    return


def set_current_step(step: int) -> None:
    global _current_step
    _current_step = int(step)


def log_event(phase: str, device: str, duration_ms: float, step: Optional[int] = None, extras: Optional[Dict[str, Any]] = None) -> None:
    return


