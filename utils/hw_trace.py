from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Dict
from threading import Lock

_log_path: Optional[Path] = None
_lock: Lock = Lock()
_current_step: int = -1


def set_log_path(path: str) -> None:
    global _log_path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _log_path = p
    # write header
    try:
        if not p.exists():
            with p.open('w', encoding='utf-8') as f:
                f.write('# LoRe Hardware Trace\n')
                f.write('# time, step, phase, device, duration_ms, extras(json)\n')
    except Exception:
        _log_path = None


def set_current_step(step: int) -> None:
    global _current_step
    _current_step = int(step)


def log_event(phase: str, device: str, duration_ms: float, step: Optional[int] = None, extras: Optional[Dict[str, Any]] = None) -> None:
    if _log_path is None:
        return
    ts = time.time()
    st = int(_current_step if step is None else step)
    line = {
        'time': ts,
        'step': st,
        'phase': phase,
        'device': device,
        'duration_ms': float(duration_ms),
        'extras': extras or {},
    }
    try:
        with _lock:
            with _log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
    except Exception:
        pass


