from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricsAggregator:
    """Lightweight training metrics aggregator for logging and summaries.

    This is a minimal implementation to satisfy usages in main loop:
      - set_training_config(cfg_dict)
      - record_step_metrics(step, metrics, rolling_success_rate)
      - record_episode(step, episode_return, episode_length, success)
      - compute_summary(total_steps) -> dict
      - save_json(summary, filename) -> path
      - save_txt(summary, filename) -> path
    """

    env_id: str
    seed: int
    delta_target: float
    save_dir: str = "runs"
    training_config: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def set_training_config(self, cfg: Dict[str, Any]) -> None:
        self.training_config = dict(cfg)

    def record_step_metrics(self, step: int, metrics: Dict[str, Any], rolling_success_rate: float = 0.0) -> None:
        rec = {
            "step": int(step),
            "metrics": dict(metrics),
            "rolling_success_rate": float(rolling_success_rate),
            "ts": time.time(),
        }
        self.steps.append(rec)

    def record_episode(self, step: int, episode_return: float, episode_length: int, success: bool) -> None:
        self.episodes.append({
            "step": int(step),
            "return": float(episode_return),
            "length": int(episode_length),
            "success": bool(success),
            "ts": time.time(),
        })

    def compute_summary(self, total_steps: int) -> Dict[str, Any]:
        # Aggregate simple statistics
        sr = 0.0
        if self.episodes:
            sr = sum(1.0 for e in self.episodes if e.get("success")) / max(len(self.episodes), 1)

        ent = None
        try:
            ents = [s["metrics"].get("policy/entropy") for s in self.steps if "policy/entropy" in s.get("metrics", {})]
            ent = sum(ents) / max(len(ents), 1) if ents else None
        except Exception:
            ent = None

        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "delta_target": self.delta_target,
            "total_steps": int(total_steps),
            "num_episodes": len(self.episodes),
            "success_rate": float(sr),
            "avg_entropy": float(ent) if ent is not None else None,
            "last_episode_return": float(self.episodes[-1]["return"]) if self.episodes else 0.0,
            "training_config": self.training_config,
        }

    def _ensure_dir(self) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        return self.save_dir

    def save_json(self, summary: Dict[str, Any], filename: str = "summary.json") -> str:
        out_dir = self._ensure_dir()
        path = os.path.join(out_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return path

    def save_txt(self, summary: Dict[str, Any], filename: str = "summary.txt") -> str:
        out_dir = self._ensure_dir()
        path = os.path.join(out_dir, filename)
        try:
            lines = [
                f"env_id: {summary.get('env_id')}",
                f"seed: {summary.get('seed')}",
                f"total_steps: {summary.get('total_steps')}",
                f"episodes: {summary.get('num_episodes')}",
                f"success_rate: {summary.get('success_rate')}",
                f"avg_entropy: {summary.get('avg_entropy')}",
                f"last_episode_return: {summary.get('last_episode_return')}",
            ]
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass
        return path




