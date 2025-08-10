from __future__ import annotations

import os
import time
from typing import Optional


class Logger:
    """Simple logger with optional TensorBoard support."""

    def __init__(self, log_dir: str = "runs/dreamer_crafter") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self._tb = SummaryWriter(log_dir=self.log_dir)
        except Exception:
            self._tb = None
        self._t0 = time.time()

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalar(tag, scalar_value, global_step)
        # Always also print lightweight logs
        if global_step % 100 == 0:
            dt = time.time() - self._t0
            print(f"[step={global_step}] {tag}={scalar_value:.4f} (t+{dt:.1f}s)")

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], global_step: int) -> None:
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

    def flush(self) -> None:
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        try:
            if self._tb is not None:
                self._tb.close()
        except Exception:
            pass

