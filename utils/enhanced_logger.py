"""Enhanced logging and metrics for LLM-integrated RL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class LLMMetricsTracker:
    """Track LLM-specific metrics during training."""
    
    def __init__(self):
        # LLM usage metrics
        self.llm_calls_total = 0
        self.llm_calls_successful = 0
        self.llm_calls_failed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Execution time tracking
        self.execution_times: List[float] = []
        self.avg_execution_time = 0.0
        
        # Feature and reward tracking
        self.shaped_rewards: List[float] = []
        self.feature_usage_count = 0
        self.mask_usage_count = 0
        self.logits_usage_count = 0
        
        # Confidence tracking
        self.confidence_scores: List[float] = []
        
        # Alpha decay tracking
        self.alpha_values: List[float] = []
        
        # Subgoal tracking  
        self.active_subgoals: Dict[str, int] = {}
        self.completed_subgoals: Dict[str, int] = {}
        
        # Performance comparison
        self.performance_with_llm: List[float] = []
        self.performance_without_llm: List[float] = []
    
    def log_llm_call(self, success: bool, execution_time: float, 
                    cache_hit: bool = False, confidence: float = 0.0,
                    used_features: bool = False, used_mask: bool = False,
                    used_logits: bool = False, r_shaped: float = 0.0) -> None:
        """Log an LLM call with metadata."""
        self.llm_calls_total += 1
        
        if success:
            self.llm_calls_successful += 1
        else:
            self.llm_calls_failed += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Execution time
        self.execution_times.append(execution_time)
        if len(self.execution_times) > 1000:  # Keep last 1000
            self.execution_times.pop(0)
        self.avg_execution_time = np.mean(self.execution_times)
        
        # Usage tracking
        if used_features:
            self.feature_usage_count += 1
        if used_mask:
            self.mask_usage_count += 1
        if used_logits:
            self.logits_usage_count += 1
        
        # Value tracking
        if confidence > 0:
            self.confidence_scores.append(confidence)
            if len(self.confidence_scores) > 1000:
                self.confidence_scores.pop(0)
        
        if r_shaped != 0:
            self.shaped_rewards.append(r_shaped)
            if len(self.shaped_rewards) > 1000:
                self.shaped_rewards.pop(0)
    
    def log_alpha_value(self, alpha: float) -> None:
        """Log current alpha value."""
        self.alpha_values.append(alpha)
        if len(self.alpha_values) > 1000:
            self.alpha_values.pop(0)
    
    def log_subgoal(self, subgoal_id: str, completed: bool = False) -> None:
        """Log subgoal activation or completion."""
        if completed:
            self.completed_subgoals[subgoal_id] = self.completed_subgoals.get(subgoal_id, 0) + 1
        else:
            self.active_subgoals[subgoal_id] = self.active_subgoals.get(subgoal_id, 0) + 1
    
    def log_performance(self, reward: float, used_llm: bool) -> None:
        """Log performance with/without LLM."""
        if used_llm:
            self.performance_with_llm.append(reward)
            if len(self.performance_with_llm) > 1000:
                self.performance_with_llm.pop(0)
        else:
            self.performance_without_llm.append(reward)
            if len(self.performance_without_llm) > 1000:
                self.performance_without_llm.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics dictionary."""
        total_calls = max(1, self.llm_calls_total)
        total_cache = max(1, self.cache_hits + self.cache_misses)
        
        metrics = {
            # Success rates
            'llm/success_rate': self.llm_calls_successful / total_calls,
            'llm/cache_hit_rate': self.cache_hits / total_cache,
            'llm/calls_per_1k_steps': self.llm_calls_total,  # Will be normalized by caller
            
            # Performance
            'llm/avg_execution_time': self.avg_execution_time,
            'llm/avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0.0,
            'llm/avg_shaped_reward': np.mean(self.shaped_rewards) if self.shaped_rewards else 0.0,
            
            # Usage statistics
            'llm/feature_usage_rate': self.feature_usage_count / total_calls,
            'llm/mask_usage_rate': self.mask_usage_count / total_calls,
            'llm/logits_usage_rate': self.logits_usage_count / total_calls,
            
            # Alpha schedule
            'llm/current_alpha': self.alpha_values[-1] if self.alpha_values else 0.0,
            'llm/alpha_trend': np.mean(np.diff(self.alpha_values[-10:])) if len(self.alpha_values) > 10 else 0.0,
            
            # Performance comparison
            'llm/performance_improvement': self._compute_performance_improvement(),
            
            # Subgoals
            'llm/active_subgoals': len(self.active_subgoals),
            'llm/completed_subgoals': sum(self.completed_subgoals.values()),
        }
        
        # Add individual subgoal metrics
        for subgoal, count in self.completed_subgoals.items():
            metrics[f'subgoals/{subgoal}'] = count
        
        return metrics
    
    def _compute_performance_improvement(self) -> float:
        """Compute performance improvement from LLM usage."""
        if not self.performance_with_llm or not self.performance_without_llm:
            return 0.0
        
        with_llm = np.mean(self.performance_with_llm[-100:])
        without_llm = np.mean(self.performance_without_llm[-100:])
        
        if without_llm == 0:
            return 0.0
        
        return (with_llm - without_llm) / abs(without_llm)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.__init__()


class EnhancedLogger:
    """Enhanced logger with LLM metrics and analysis."""
    
    def __init__(self, log_dir: str = "runs/enhanced_dreamer"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = None
        if SummaryWriter:
            self.writer = SummaryWriter(str(self.log_dir))
        
        # Metrics tracker
        self.llm_metrics = LLMMetricsTracker()
        
        # Episode tracking
        self.episode_data: List[Dict[str, Any]] = []
        self.current_episode = {}
        
        # JSON log file
        self.json_log_path = self.log_dir / "metrics.jsonl"
        
        # Performance history
        self.training_history: Dict[str, List[float]] = {
            'episode_returns': [],
            'episode_lengths': [],
            'success_rates': [],
            'llm_usage_rates': [],
        }
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, value_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        if self.writer:
            self.writer.add_scalars(tag, value_dict, step)
    
    def log_histogram(self, tag: str, values: Union[List, np.ndarray], step: int) -> None:
        """Log histogram of values."""
        if self.writer:
            if isinstance(values, list):
                values = np.array(values)
            self.writer.add_histogram(tag, values, step)
    
    def log_llm_metrics(self, step: int) -> None:
        """Log all LLM metrics."""
        metrics = self.llm_metrics.get_metrics()
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(key, float(value), step)
    
    def log_episode_end(self, episode_return: float, episode_length: int, 
                       success: bool, llm_usage: float, step: int,
                       additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log episode completion."""
        # Update history
        self.training_history['episode_returns'].append(episode_return)
        self.training_history['episode_lengths'].append(episode_length)
        self.training_history['success_rates'].append(1.0 if success else 0.0)
        self.training_history['llm_usage_rates'].append(llm_usage)
        
        # Maintain history size
        max_history = 1000
        for key in self.training_history:
            if len(self.training_history[key]) > max_history:
                self.training_history[key].pop(0)
        
        # Log to TensorBoard
        self.log_scalar('episode/return', episode_return, step)
        self.log_scalar('episode/length', episode_length, step)
        self.log_scalar('episode/success', 1.0 if success else 0.0, step)
        self.log_scalar('episode/llm_usage', llm_usage, step)
        
        # Log moving averages
        if len(self.training_history['episode_returns']) >= 10:
            avg_return = np.mean(self.training_history['episode_returns'][-100:])
            avg_success = np.mean(self.training_history['success_rates'][-100:])
            avg_llm_usage = np.mean(self.training_history['llm_usage_rates'][-100:])
            
            self.log_scalar('episode/avg_return_100', avg_return, step)
            self.log_scalar('episode/avg_success_100', avg_success, step)
            self.log_scalar('episode/avg_llm_usage_100', avg_llm_usage, step)
        
        # Log additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                if isinstance(value, (int, float)):
                    self.log_scalar(f'episode/{key}', float(value), step)
        
        # JSON log entry
        log_entry = {
            'step': step,
            'timestamp': time.time(),
            'episode_return': episode_return,
            'episode_length': episode_length,
            'success': success,
            'llm_usage': llm_usage,
            **(additional_metrics or {}),
        }
        
        self._write_json_log(log_entry)
    
    def log_crafter_achievements(self, achievements: Dict[str, Any], step: int) -> None:
        """Log Crafter-specific achievements."""
        for achievement, value in achievements.items():
            if isinstance(value, bool):
                self.log_scalar(f'crafter/{achievement}', 1.0 if value else 0.0, step)
            elif isinstance(value, (int, float)):
                self.log_scalar(f'crafter/{achievement}', float(value), step)
    
    def log_training_phase(self, phase: str, metrics: Dict[str, float], step: int) -> None:
        """Log training phase metrics (collection, update, etc.)."""
        for key, value in metrics.items():
            self.log_scalar(f'{phase}/{key}', value, step)
    
    def create_summary_report(self, step: int) -> Dict[str, Any]:
        """Create comprehensive summary report."""
        llm_metrics = self.llm_metrics.get_metrics()
        
        # Performance summary
        recent_returns = self.training_history['episode_returns'][-100:]
        recent_success = self.training_history['success_rates'][-100:]
        recent_llm_usage = self.training_history['llm_usage_rates'][-100:]
        
        summary = {
            'step': step,
            'timestamp': time.time(),
            
            # Performance
            'performance': {
                'avg_return': float(np.mean(recent_returns)) if recent_returns else 0.0,
                'max_return': float(np.max(recent_returns)) if recent_returns else 0.0,
                'success_rate': float(np.mean(recent_success)) if recent_success else 0.0,
                'avg_episode_length': float(np.mean(self.training_history['episode_lengths'][-100:])),
            },
            
            # LLM integration
            'llm_integration': llm_metrics,
            
            # Training efficiency
            'efficiency': {
                'llm_usage_rate': float(np.mean(recent_llm_usage)) if recent_llm_usage else 0.0,
                'cache_efficiency': llm_metrics.get('llm/cache_hit_rate', 0.0),
                'avg_execution_time': llm_metrics.get('llm/avg_execution_time', 0.0),
            },
            
            # Model state
            'model_state': {
                'current_alpha': llm_metrics.get('llm/current_alpha', 0.0),
                'performance_improvement': llm_metrics.get('llm/performance_improvement', 0.0),
                'active_subgoals': llm_metrics.get('llm/active_subgoals', 0),
            },
        }
        
        # Write detailed report
        report_path = self.log_dir / f"summary_report_{step}.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _write_json_log(self, entry: Dict[str, Any]) -> None:
        """Write JSON log entry."""
        with open(self.json_log_path, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
    
    def flush(self) -> None:
        """Flush all logs."""
        if self.writer:
            self.writer.flush()
    
    def close(self) -> None:
        """Close logger."""
        if self.writer:
            self.writer.close()


def create_comparison_dashboard(log_dirs: List[str], output_path: str) -> None:
    """Create comparison dashboard between different runs."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LLM-Enhanced RL Comparison Dashboard', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, log_dir in enumerate(log_dirs[:5]):  # Limit to 5 runs
        metrics_file = Path(log_dir) / "metrics.jsonl"
        if not metrics_file.exists():
            continue
        
        # Load metrics
        data = []
        with open(metrics_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        if not data:
            continue
        
        steps = [entry['step'] for entry in data]
        returns = [entry['episode_return'] for entry in data]
        success = [entry['success'] for entry in data]
        llm_usage = [entry['llm_usage'] for entry in data]
        
        label = f'Run {i+1}'
        color = colors[i % len(colors)]
        
        # Episode returns
        axes[0, 0].plot(steps, returns, color=color, label=label, alpha=0.7)
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Return')
        
        # Success rate (moving average)
        if len(success) >= 100:
            success_ma = [np.mean(success[max(0, j-100):j+1]) for j in range(len(success))]
            axes[0, 1].plot(steps, success_ma, color=color, label=label)
        axes[0, 1].set_title('Success Rate (100-episode MA)')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Success Rate')
        
        # LLM usage
        axes[0, 2].plot(steps, llm_usage, color=color, label=label, alpha=0.7)
        axes[0, 2].set_title('LLM Usage Rate')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Usage Rate')
    
    # Add legends
    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison dashboard saved to {output_path}")