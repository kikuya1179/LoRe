"""No-Op enhanced logger. All methods are inert to disable logging output."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


class LLMMetricsTracker:
    """No-Op metrics tracker."""

    def __init__(self) -> None:
        return

    def log_llm_call(self, *args, **kwargs) -> None:
        return

    def log_alpha_value(self, *args, **kwargs) -> None:
        return

    def log_subgoal(self, *args, **kwargs) -> None:
        return

    def log_performance(self, *args, **kwargs) -> None:
        return

    def get_metrics(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        return


class EnhancedLogger:
    """No-Op enhanced logger that preserves API surface."""

    def __init__(self, log_dir: str = "runs/enhanced_dreamer") -> None:
        self.log_dir = log_dir
        self.llm_metrics = LLMMetricsTracker()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        return

    def log_scalars(self, tag: str, value_dict: Dict[str, float], step: int) -> None:
        return

    def log_histogram(self, tag: str, values: Union[List, Any], step: int) -> None:
        return

    def log_llm_metrics(self, step: int) -> None:
        return

    def log_episode_end(
        self,
        episode_return: float,
        episode_length: int,
        success: bool,
        llm_usage: float,
        step: int,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        return

    def log_crafter_achievements(self, achievements: Dict[str, Any], step: int) -> None:
        return

    def log_training_phase(self, phase: str, metrics: Dict[str, float], step: int) -> None:
        return

    def create_summary_report(self, step: int) -> Dict[str, Any]:
        return {}

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


def create_comparison_dashboard(log_dirs: List[str], output_path: str) -> None:
    return