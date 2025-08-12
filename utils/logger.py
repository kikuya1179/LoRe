from __future__ import annotations

from typing import Optional, Dict


class Logger:
    """No-Op logger to disable all logging without breaking interfaces."""

    def __init__(self, log_dir: str = "runs/dreamer_crafter", *, allowed_tags: Optional[set[str]] = None) -> None:
        self.log_dir = log_dir
        self._allowed = allowed_tags
        self._current_step = 0

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:  # noqa: D401
        return

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int) -> None:
        return

    def maybe_step(self, global_step: int) -> None:
        return

    def flush(self, force: bool = False) -> None:
        return

    def set_current_step(self, step: int) -> None:
        self._current_step = int(step)

    def close(self) -> None:
        return

