from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class AggregatedSummary:
    env_id: str
    seed: int
    total_steps: int
    wall_time_sec: float
    # Minimal set
    success_rate: float
    auc_normalized: float
    t80_steps: Optional[int]
    t95_steps: Optional[int]
    returns_mean: float
    episode_length_mean: float
    # Extended set
    time_to_first_success_steps: Optional[int]
    successes_per_1k: float
    # LoRe specific (may be None when disabled)
    beta_mean: Optional[float]
    beta_std: Optional[float]
    kl_mean: Optional[float]
    kl_exceed_rate: Optional[float]
    lambda_kl: Optional[float]
    # LLM costs (optional)
    llm_calls_used: Optional[int]
    llm_budget_remaining: Optional[int]
    llm_cache_hit_rate: Optional[float]


class MetricsAggregator:
    def __init__(self, env_id: str, seed: int, delta_target: float, save_dir: str = "runs") -> None:
        self.env_id = env_id
        self.seed = seed
        self.delta_target = float(delta_target)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Time
        self.t0 = time.time()

        # Episode-level
        self.episode_end_steps: List[int] = []
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_success_flags: List[int] = []

        # Rolling SR over steps (sampled)
        self.sr_points: List[Tuple[int, float]] = []  # (step, sr_rolling)

        # Success events (step indices)
        self.success_steps: List[int] = []

        # LoRe-related
        self.kl_samples: List[float] = []
        self.beta_samples: List[float] = []

        # LLM stats (last snapshot)
        self.llm_stats: Dict[str, float] = {}

        # Skill metrics (last snapshot)
        self.skill_last: Dict[str, float] = {}

        # Training config snapshot (optional)
        self.training_config: Dict[str, str | float | int] = {}

    def record_episode(self, step: int, episode_return: float, episode_length: int, success: bool) -> None:
        self.episode_end_steps.append(step)
        self.episode_returns.append(float(episode_return))
        self.episode_lengths.append(int(episode_length))
        self.episode_success_flags.append(1 if success else 0)
        if success:
            self.success_steps.append(step)

    def record_step_metrics(self, step: int, metrics: Dict[str, float], rolling_success_rate: float) -> None:
        # Rolling success rate point for AUC/T_x
        self.sr_points.append((step, float(rolling_success_rate)))

        # LoRe
        if 'loss/kl_divergence' in metrics:
            try:
                self.kl_samples.append(float(metrics['loss/kl_divergence']))
            except Exception:
                pass
        if 'beta/mean' in metrics:
            try:
                self.beta_samples.append(float(metrics['beta/mean']))
            except Exception:
                pass

        # Skill metrics snapshot
        for k in (
            'skill/pickup_key_rate','skill/door_toggle_rate','skill/door_unlock_open_rate',
            'skill/has_key_ratio','skill/invalid_action_ratio',
            'skill/dist_med_ag_key','skill/dist_med_key_door','skill/dist_med_door_goal',
            'policy/entropy_inst','policy/prob_max_inst'
        ):
            if k in metrics:
                try:
                    self.skill_last[k] = float(metrics[k])
                except Exception:
                    pass

    def set_llm_stats(self, stats: Dict[str, float]) -> None:
        self.llm_stats = stats.copy()

    def set_training_config(self, cfg: Dict[str, str | float | int]) -> None:
        self.training_config = cfg.copy()

    def _auc_normalized(self, total_steps: int) -> float:
        if not self.sr_points:
            return 0.0
        # Trapezoidal integral over sampled SR points
        pts = sorted(self.sr_points, key=lambda x: x[0])
        area = 0.0
        prev_s, prev_sr = pts[0]
        for s, sr in pts[1:]:
            ds = max(0, s - prev_s)
            area += 0.5 * (sr + prev_sr) * ds
            prev_s, prev_sr = s, sr
        if total_steps <= 0:
            return 0.0
        return float(area) / float(total_steps)

    def _time_to_x(self, x: float) -> Optional[int]:
        for s, sr in sorted(self.sr_points, key=lambda x: x[0]):
            if sr >= x:
                return s
        return None

    def _successes_per_1k(self, total_steps: int) -> float:
        if total_steps <= 0:
            return 0.0
        return 1000.0 * (len(self.success_steps) / max(1.0, float(total_steps)))

    def compute_summary(self, total_steps: int) -> AggregatedSummary:
        wall_time = time.time() - self.t0

        sr = 0.0
        if self.sr_points:
            sr = self.sr_points[-1][1]

        returns_mean = sum(self.episode_returns) / max(1, len(self.episode_returns))
        ep_len_mean = sum(self.episode_lengths) / max(1, len(self.episode_lengths))

        t_first = self.success_steps[0] if self.success_steps else None
        t80 = self._time_to_x(0.8)
        t95 = self._time_to_x(0.95)
        auc_n = self._auc_normalized(total_steps)
        succ_per_1k = self._successes_per_1k(total_steps)

        # LoRe metrics
        beta_mean = (sum(self.beta_samples) / len(self.beta_samples)) if self.beta_samples else None
        beta_std = None
        if self.beta_samples:
            m = beta_mean or 0.0
            var = sum((b - m) ** 2 for b in self.beta_samples) / max(1, len(self.beta_samples) - 1)
            beta_std = var ** 0.5
        kl_mean = (sum(self.kl_samples) / len(self.kl_samples)) if self.kl_samples else None
        kl_exceed_rate = None
        if self.kl_samples:
            exceed = sum(1 for k in self.kl_samples if k > self.delta_target)
            kl_exceed_rate = exceed / max(1, len(self.kl_samples))

        lam_kl = None
        # can be filled from live metrics: uncertainty_gate/lambda_kl if provided

        # LLM stats
        llm_calls_used = int(self.llm_stats.get('llm_calls_used', 0)) if self.llm_stats else None
        llm_budget_remaining = int(self.llm_stats.get('llm_budget_remaining', 0)) if self.llm_stats else None
        llm_cache_hit_rate = float(self.llm_stats.get('llm_cache_hit_rate', 0.0)) if self.llm_stats else None

        return AggregatedSummary(
            env_id=self.env_id,
            seed=self.seed,
            total_steps=total_steps,
            wall_time_sec=wall_time,
            success_rate=sr,
            auc_normalized=auc_n,
            t80_steps=t80,
            t95_steps=t95,
            returns_mean=returns_mean,
            episode_length_mean=ep_len_mean,
            time_to_first_success_steps=t_first,
            successes_per_1k=succ_per_1k,
            beta_mean=beta_mean,
            beta_std=beta_std,
            kl_mean=kl_mean,
            kl_exceed_rate=kl_exceed_rate,
            lambda_kl=lam_kl,
            llm_calls_used=llm_calls_used,
            llm_budget_remaining=llm_budget_remaining,
            llm_cache_hit_rate=llm_cache_hit_rate,
        )

    def save_json(self, summary: AggregatedSummary, filename: str) -> str:
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
        return path

    def save_txt(self, summary: AggregatedSummary, filename: str) -> str:
        path = os.path.join(self.save_dir, filename)
        lines = [
            f"env={summary.env_id} seed={summary.seed} steps={summary.total_steps} time={summary.wall_time_sec:.1f}s",
            f"SR={summary.success_rate:.3f} AUC_N={summary.auc_normalized:.3f} T80={summary.t80_steps} T95={summary.t95_steps}",
            f"Return_mean={summary.returns_mean:.3f} EpLen_mean={summary.episode_length_mean:.1f}",
            f"FirstSuccess={summary.time_to_first_success_steps} Successes/1k={summary.successes_per_1k:.2f}",
        ]
        # Skill metrics block
        if self.skill_last:
            lines.append("-- skills --")
            def _g(key, fmt=".3f", default=None):
                v = self.skill_last.get(key, default)
                return (f"{v:{fmt}}" if isinstance(v, (int,float)) else str(v)) if v is not None else "NA"
            lines.append(
                "pickup="+_g('skill/pickup_key_rate')+" toggle="+_g('skill/door_toggle_rate')+
                " unlockOpen="+_g('skill/door_unlock_open_rate')+" hasKey="+_g('skill/has_key_ratio')+
                " invalid="+_g('skill/invalid_action_ratio')
            )
            lines.append(
                "d_ag->key="+_g('skill/dist_med_ag_key', ".1f")+
                " d_key->door="+_g('skill/dist_med_key_door', ".1f")+
                " d_door->goal="+_g('skill/dist_med_door_goal', ".1f")
            )
            lines.append(
                "entropy_inst="+_g('policy/entropy_inst')+" prob_max_inst="+_g('policy/prob_max_inst')
            )
        if summary.beta_mean is not None:
            lines.append(f"beta_mean={summary.beta_mean:.3f} beta_std={summary.beta_std or 0.0:.3f}")
        if summary.kl_mean is not None:
            lines.append(f"kl_mean={summary.kl_mean:.3f} kl_exceed_rate={summary.kl_exceed_rate or 0.0:.3f}")
        if summary.llm_calls_used is not None:
            lines.append(
                f"llm_calls={summary.llm_calls_used} budget_remain={summary.llm_budget_remaining} cache_hit={summary.llm_cache_hit_rate}"
            )
        # Training config snapshot
        if self.training_config:
            lines.append("-- train_config --")
            for k, v in self.training_config.items():
                lines.append(f"{k}={v}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        return path


