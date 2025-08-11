from __future__ import annotations

import os
import time
from typing import Optional


class Logger:
    """Simple logger with optional TensorBoard support and tag filtering.

    Only tags in `allowed_tags` will be emitted. If `allowed_tags` is None, all
    tags are emitted.
    """

    def __init__(self, log_dir: str = "runs/dreamer_crafter", *, allowed_tags: Optional[set[str]] = None) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self._tb = SummaryWriter(log_dir=self.log_dir)
        except Exception:
            self._tb = None
        self._t0 = time.time()
        # Whitelist for tags
        self._allowed = allowed_tags

    def _is_allowed(self, tag: str) -> bool:
        if self._allowed is None:
            return True
        return tag in self._allowed

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if not self._is_allowed(tag):
            return
        if self._tb is not None:
            self._tb.add_scalar(tag, scalar_value, global_step)
        # Always also print lightweight logs
        if global_step % 100 == 0:
            dt = time.time() - self._t0
            print(f"[step={global_step}] {tag}={scalar_value:.4f} (t+{dt:.1f}s)")

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], global_step: int) -> None:
        if not tag_scalar_dict:
            return
        # Filter by allowed tags
        if self._allowed is not None:
            tag_scalar_dict = {k: v for k, v in tag_scalar_dict.items() if self._is_allowed(f"{main_tag}/{k}")}
            if not tag_scalar_dict:
                return
        if self._tb is not None:
            try:
                self._tb.add_scalars(main_tag, tag_scalar_dict, global_step)
            except Exception:
                # Fallback: log individually
                for k, v in tag_scalar_dict.items():
                    self._tb.add_scalar(f"{main_tag}/{k}", v, global_step)
        # Print compact summary occasionally
        if global_step % 100 == 0:
            try:
                parts = ", ".join(f"{k}={v:.3f}" for k, v in list(tag_scalar_dict.items())[:4])
                dt = time.time() - self._t0
                print(f"[step={global_step}] {main_tag}: {parts} ... (t+{dt:.1f}s)")
            except Exception:
                pass

    def flush(self, force: bool = False) -> None:
        # Rate-limit flush to avoid I/O bottleneck
        if not force:
            if not hasattr(self, '_last_flush_step'):
                self._last_flush_step = 0
            # Only flush every 500 steps by default
            if hasattr(self, '_current_step') and (self._current_step - self._last_flush_step) < 500:
                return
            self._last_flush_step = getattr(self, '_current_step', 0)
        if self._tb is not None:
            self._tb.flush()
    
    def set_current_step(self, step: int) -> None:
        """Update current step for flush rate limiting"""
        self._current_step = step

    def close(self) -> None:
        try:
            if self._tb is not None:
                self._tb.close()
        except Exception:
            pass

