from __future__ import annotations

from typing import Dict, List


class HealthMonitor:
    """Lightweight training health monitor.

    Detects stalls/divergence based on simple metric thresholds.
    """

    def __init__(self) -> None:
        self.last_entropy = None
        self.low_entropy_ticks = 0
        self.high_kl_ticks = 0
        self.alerts: List[str] = []

    def update(self, step: int, metrics: Dict[str, float], **kwargs) -> List[str]:
        # Backward/forward compatible kwargs (accept any extra)
        warmup_steps: int = int(kwargs.get('warmup_steps', 0) or 0)
        verbose: bool = bool(kwargs.get('verbose', False))
        self.alerts.clear()

        # Warmup period: suppress alerts
        if step < warmup_steps:
            return []

        entropy = metrics.get('policy/entropy', None)
        if entropy is not None:
            if entropy < 0.05:
                self.low_entropy_ticks += 1
            else:
                self.low_entropy_ticks = 0
            if self.low_entropy_ticks > 200:
                self.alerts.append('Entropy too low for prolonged steps (policy collapse risk)')

        kl = metrics.get('loss/kl_divergence', None)
        if kl is not None and kl > 1.0:
            self.high_kl_ticks += 1
            if self.high_kl_ticks > 50:
                self.alerts.append('KL divergence persistently high (prior too strong)')
        else:
            self.high_kl_ticks = 0

        return list(self.alerts) if verbose else []


def create_health_dashboard(monitor: HealthMonitor, recent_metrics: List[Dict[str, float]]) -> str:
    if not recent_metrics:
        return "(no data)"
    last = recent_metrics[-1]
    lines = [
        "==== Health Dashboard ====",
        f"entropy={last.get('policy/entropy', 0):.3f}",
        f"kl={last.get('loss/kl_divergence', 0):.3f}",
        f"adv_mean={last.get('advantage/mean', 0):.3f}",
    ]
    return "\n".join(lines)

"""
Health Monitoring and Early Anomaly Detection for LoRe Training
"""

import time
import warnings
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np

import torch


@dataclass 
class HealthAlert:
    """Health monitoring alert."""
    timestamp: float
    severity: str  # 'warning', 'error', 'critical'
    component: str  # 'policy', 'value', 'world_model', 'llm', 'uncertainty_gate'
    metric: str
    current_value: float
    threshold: float
    message: str
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.component}.{self.metric}: {self.message}"


class HealthMonitor:
    """Monitors training health and detects anomalies early."""
    
    def __init__(self, window_size: int = 100, alert_cooldown: int = 50):
        self.window_size = window_size
        self.alert_cooldown = alert_cooldown
        
        # Metric history buffers
        self.history: Dict[str, deque] = {}
        self.alerts: List[HealthAlert] = []
        self.last_alert_step: Dict[str, int] = {}
        
        # Thresholds (will be auto-calibrated)
        self.thresholds = {
            # Loss thresholds
            'loss/policy': {'max': 10.0, 'spike_factor': 3.0},
            'loss/value': {'max': 5.0, 'spike_factor': 3.0}, 
            'loss/model_recon': {'max': 1.0, 'spike_factor': 2.0},
            
            # Gradient thresholds
            'optim/grad_global_norm': {'max': 20.0, 'spike_factor': 5.0},
            
            # Policy health
            'policy/entropy': {'min': 0.1, 'max': 4.0},
            'value/explained_variance': {'min': 0.0},
            
            # LLM health
            'llm_cache_hit_rate': {'min': 0.1},
            'llm_trigger_invalid_response': {'max_rate': 0.3},
            
            # Uncertainty gate health  
            'uncertainty_gate/avg_beta': {'min': 0.0, 'max': 1.0},
            'uncertainty_gate/avg_kl': {'max': 2.0, 'spike_factor': 3.0},
            
            # Performance indicators
            'success_rate': {'plateau_frames': 2000, 'min_improvement': 0.01},
        }
        
        # Auto-calibration state
        self.calibration_steps = 0
        self.calibration_target = 200  # Calibrate thresholds for first 200 steps
        
    def update(self, step: int, metrics: Dict[str, float], **kwargs) -> List[HealthAlert]:
        """Update monitoring with new metrics and return any new alerts.
        Accepts optional kwargs (e.g., warmup_steps, verbose) for compatibility.
        """
        warmup_steps = int(kwargs.get('warmup_steps', 0) or 0)
        # Suppress alerts during warmup period
        if step < warmup_steps:
            return []
        new_alerts = []
        
        # Store metrics in history
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = deque(maxlen=self.window_size)
            self.history[key].append(value)
        
        # Auto-calibrate thresholds during initial training
        if self.calibration_steps < self.calibration_target:
            self._auto_calibrate(metrics)
            self.calibration_steps += 1
            return new_alerts
        
        # Check all health conditions
        new_alerts.extend(self._check_loss_health(step, metrics))
        new_alerts.extend(self._check_gradient_health(step, metrics))
        new_alerts.extend(self._check_policy_health(step, metrics))
        new_alerts.extend(self._check_llm_health(step, metrics))
        new_alerts.extend(self._check_uncertainty_health(step, metrics))
        new_alerts.extend(self._check_performance_health(step, metrics))
        
        # Store alerts and apply cooldown
        for alert in new_alerts:
            alert_key = f"{alert.component}.{alert.metric}"
            if (alert_key not in self.last_alert_step or 
                step - self.last_alert_step[alert_key] >= self.alert_cooldown):
                self.alerts.append(alert)
                self.last_alert_step[alert_key] = step
            
        return new_alerts
    
    def _auto_calibrate(self, metrics: Dict[str, float]):
        """Auto-calibrate thresholds based on initial training."""
        for key, value in metrics.items():
            if key in self.history and len(self.history[key]) >= 10:
                values = list(self.history[key])
                
                # Set max thresholds based on observed data
                if key in ['loss/policy', 'loss/value', 'loss/model_recon']:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    self.thresholds[key]['max'] = mean_val + 3 * std_val
                    
                elif key == 'optim/grad_global_norm':
                    max_val = np.max(values)
                    self.thresholds[key]['max'] = max_val * 2.0
                    
                elif key == 'policy/entropy':
                    min_val = np.min(values)
                    max_val = np.max(values)
                    self.thresholds[key]['min'] = max(0.01, min_val * 0.5)
                    self.thresholds[key]['max'] = max_val * 2.0
    
    def _check_loss_health(self, step: int, metrics: Dict[str, float]) -> List[HealthAlert]:
        """Check for loss-related health issues."""
        alerts = []
        
        for loss_key in ['loss/policy', 'loss/value', 'loss/model_recon']:
            if loss_key not in metrics:
                continue
                
            current_val = metrics[loss_key]
            threshold_config = self.thresholds.get(loss_key, {})
            
            # Check for absolute threshold violations
            max_threshold = threshold_config.get('max', float('inf'))
            if current_val > max_threshold:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='error',
                    component='training',
                    metric=loss_key,
                    current_value=current_val,
                    threshold=max_threshold,
                    message=f"Loss {current_val:.4f} exceeds maximum threshold {max_threshold:.4f}"
                ))
            
            # Check for sudden spikes
            if loss_key in self.history and len(self.history[loss_key]) >= 10:
                recent_avg = np.mean(list(self.history[loss_key])[-10:])
                spike_factor = threshold_config.get('spike_factor', 3.0)
                if current_val > recent_avg * spike_factor:
                    alerts.append(HealthAlert(
                        timestamp=time.time(),
                        severity='warning',
                        component='training',
                        metric=loss_key,
                        current_value=current_val,
                        threshold=recent_avg * spike_factor,
                        message=f"Loss spike detected: {current_val:.4f} vs recent avg {recent_avg:.4f}"
                    ))
        
        return alerts
    
    def _check_gradient_health(self, step: int, metrics: Dict[str, float]) -> List[HealthAlert]:
        """Check gradient health."""
        alerts = []
        
        grad_norm_key = 'optim/grad_global_norm'
        if grad_norm_key in metrics:
            grad_norm = metrics[grad_norm_key]
            max_threshold = self.thresholds[grad_norm_key]['max']
            
            if grad_norm > max_threshold:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='warning',
                    component='gradients',
                    metric=grad_norm_key,
                    current_value=grad_norm,
                    threshold=max_threshold,
                    message=f"Gradient norm {grad_norm:.3f} indicates potential instability"
                ))
            
            # Check for gradient explosion pattern
            if grad_norm_key in self.history and len(self.history[grad_norm_key]) >= 5:
                recent_norms = list(self.history[grad_norm_key])[-5:]
                if all(norm > 5.0 for norm in recent_norms):
                    alerts.append(HealthAlert(
                        timestamp=time.time(),
                        severity='error',
                        component='gradients',
                        metric=grad_norm_key,
                        current_value=grad_norm,
                        threshold=5.0,
                        message=f"Gradient explosion pattern detected"
                    ))
        
        return alerts
    
    def _check_policy_health(self, step: int, metrics: Dict[str, float]) -> List[HealthAlert]:
        """Check policy health indicators."""
        alerts = []
        
        # Entropy checks
        entropy_key = 'policy/entropy'
        if entropy_key in metrics:
            entropy = metrics[entropy_key]
            min_entropy = self.thresholds[entropy_key]['min']
            max_entropy = self.thresholds[entropy_key]['max']
            
            if entropy < min_entropy:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='warning',
                    component='policy',
                    metric=entropy_key,
                    current_value=entropy,
                    threshold=min_entropy,
                    message=f"Policy entropy {entropy:.3f} too low - policy may be deterministic"
                ))
            elif entropy > max_entropy:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='warning',
                    component='policy',
                    metric=entropy_key,
                    current_value=entropy,
                    threshold=max_entropy,
                    message=f"Policy entropy {entropy:.3f} too high - policy may be random"
                ))
        
        # Value function checks
        ev_key = 'value/explained_variance'
        if ev_key in metrics:
            explained_var = metrics[ev_key]
            min_ev = self.thresholds[ev_key]['min']
            
            if explained_var < min_ev and step > 1000:  # Allow initial learning
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='warning',
                    component='value',
                    metric=ev_key,
                    current_value=explained_var,
                    threshold=min_ev,
                    message=f"Value function not learning - explained variance {explained_var:.3f}"
                ))
        
        return alerts
    
    def _check_llm_health(self, step: int, metrics: Dict[str, float]) -> List[HealthAlert]:
        """Check LLM integration health."""
        alerts = []
        
        # Cache hit rate
        cache_key = 'llm_cache_hit_rate'
        if cache_key in metrics:
            hit_rate = metrics[cache_key]
            min_hit_rate = self.thresholds[cache_key]['min']
            
            if hit_rate < min_hit_rate and step > 500:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='warning',
                    component='llm',
                    metric=cache_key,
                    current_value=hit_rate,
                    threshold=min_hit_rate,
                    message=f"LLM cache hit rate {hit_rate:.3f} very low - check diversity"
                ))
        
        # Invalid response rate
        invalid_key = 'llm_trigger_invalid_response'
        if invalid_key in metrics and step > 100:
            invalid_count = metrics[invalid_key]
            if invalid_key in self.history and len(self.history[invalid_key]) >= 10:
                recent_invalids = list(self.history[invalid_key])
                invalid_rate = (recent_invalids[-1] - recent_invalids[-10]) / 10.0
                max_rate = self.thresholds[invalid_key]['max_rate']
                
                if invalid_rate > max_rate:
                    alerts.append(HealthAlert(
                        timestamp=time.time(),
                        severity='error',
                        component='llm',
                        metric=invalid_key,
                        current_value=invalid_rate,
                        threshold=max_rate,
                        message=f"High LLM failure rate {invalid_rate:.3f} - check API/prompt"
                    ))
        
        return alerts
    
    def _check_uncertainty_health(self, step: int, metrics: Dict[str, float]) -> List[HealthAlert]:
        """Check uncertainty gate health."""
        alerts = []
        
        # Beta range checks
        beta_key = 'uncertainty_gate/avg_beta'
        if beta_key in metrics:
            avg_beta = metrics[beta_key]
            min_beta = self.thresholds[beta_key]['min']
            max_beta = self.thresholds[beta_key]['max']
            
            if avg_beta < min_beta and step > 1000:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='info',
                    component='uncertainty_gate',
                    metric=beta_key,
                    current_value=avg_beta,
                    threshold=min_beta,
                    message=f"Beta consistently low {avg_beta:.3f} - LLM rarely used"
                ))
            elif avg_beta > max_beta:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='warning',
                    component='uncertainty_gate',
                    metric=beta_key,
                    current_value=avg_beta,
                    threshold=max_beta,
                    message=f"Beta too high {avg_beta:.3f} - over-reliance on LLM"
                ))
        
        # KL divergence checks
        kl_key = 'uncertainty_gate/avg_kl'
        if kl_key in metrics:
            avg_kl = metrics[kl_key]
            max_kl = self.thresholds[kl_key]['max']
            
            if avg_kl > max_kl:
                alerts.append(HealthAlert(
                    timestamp=time.time(),
                    severity='warning',
                    component='uncertainty_gate',
                    metric=kl_key,
                    current_value=avg_kl,
                    threshold=max_kl,
                    message=f"KL divergence {avg_kl:.3f} too high - policy diverging"
                ))
        
        return alerts
    
    def _check_performance_health(self, step: int, metrics: Dict[str, float]) -> List[HealthAlert]:
        """Check overall performance health."""
        alerts = []
        
        # Success rate plateau detection
        success_key = 'success_rate'
        if success_key in metrics and success_key in self.history:
            current_sr = metrics[success_key]
            plateau_config = self.thresholds[success_key]
            plateau_frames = plateau_config['plateau_frames']
            min_improvement = plateau_config['min_improvement']
            
            if len(self.history[success_key]) >= plateau_frames // 10:  # Check every 10th of plateau window
                recent_srs = list(self.history[success_key])[-plateau_frames//10:]
                if len(recent_srs) >= 2:
                    improvement = recent_srs[-1] - recent_srs[0]
                    if improvement < min_improvement:
                        alerts.append(HealthAlert(
                            timestamp=time.time(),
                            severity='info',
                            component='performance',
                            metric=success_key,
                            current_value=current_sr,
                            threshold=min_improvement,
                            message=f"Performance plateau: {improvement:.3f} improvement in recent window"
                        ))
        
        return alerts
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]  # Last hour
        
        severity_counts = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
        
        component_issues = {}
        for alert in recent_alerts:
            if alert.component not in component_issues:
                component_issues[alert.component] = 0
            component_issues[alert.component] += 1
        
        # Overall health score (0-100)
        health_score = 100
        health_score -= severity_counts['warning'] * 5
        health_score -= severity_counts['error'] * 15
        health_score -= severity_counts['critical'] * 30
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'recent_alerts': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'component_issues': component_issues,
            'calibration_complete': self.calibration_steps >= self.calibration_target
        }
    
    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on recent alerts."""
        recommendations = []
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 1800]  # Last 30 min
        
        # Group by component and issue type
        component_alerts = {}
        for alert in recent_alerts:
            if alert.component not in component_alerts:
                component_alerts[alert.component] = []
            component_alerts[alert.component].append(alert)
        
        # Generate specific recommendations
        for component, alerts in component_alerts.items():
            if component == 'training':
                if any('spike' in a.message for a in alerts):
                    recommendations.append("Consider reducing learning rate or increasing gradient clipping")
                if any('exceeds maximum' in a.message for a in alerts):
                    recommendations.append("Check for NaN/inf values or reduce batch size")
            
            elif component == 'policy':
                if any('entropy' in a.metric for a in alerts):
                    recommendations.append("Adjust entropy coefficient to balance exploration/exploitation")
            
            elif component == 'llm':
                if any('cache_hit_rate' in a.metric for a in alerts):
                    recommendations.append("Increase state diversity or check LLM trigger conditions")
                if any('invalid_response' in a.metric for a in alerts):
                    recommendations.append("Check LLM API status and prompt format")
            
            elif component == 'uncertainty_gate':
                if any('beta' in a.metric for a in alerts):
                    recommendations.append("Tune uncertainty thresholds and beta range")
                if any('kl' in a.metric for a in alerts):
                    recommendations.append("Adjust KL target or learning rate")
        
        return recommendations[:5]  # Top 5 recommendations


def create_health_dashboard(monitor: HealthMonitor, metrics_history: List[Dict[str, float]]) -> str:
    """Create a simple text-based health dashboard."""
    
    dashboard = []
    dashboard.append("="*60)
    dashboard.append("           LoRe Training Health Dashboard")
    dashboard.append("="*60)
    
    # Health summary
    summary = monitor.get_health_summary()
    dashboard.append(f"\nOverall Health Score: {summary['health_score']}/100")
    
    if summary['health_score'] >= 80:
        status = "🟢 HEALTHY"
    elif summary['health_score'] >= 60:
        status = "🟡 CAUTION"
    else:
        status = "🔴 ISSUES DETECTED"
    
    dashboard.append(f"Status: {status}")
    dashboard.append(f"Recent Alerts: {summary['recent_alerts']}")
    
    # Alert breakdown
    if summary['recent_alerts'] > 0:
        dashboard.append(f"\nAlert Breakdown:")
        for severity, count in summary['severity_breakdown'].items():
            if count > 0:
                dashboard.append(f"  {severity.upper()}: {count}")
    
    # Component status
    if summary['component_issues']:
        dashboard.append(f"\nComponent Issues:")
        for component, count in summary['component_issues'].items():
            dashboard.append(f"  {component}: {count} alerts")
    
    # Recent metrics (last 5 points)
    if metrics_history:
        dashboard.append(f"\nRecent Metrics Trend:")
        recent_metrics = metrics_history[-5:]
        
        key_metrics = ['success_rate', 'policy/entropy', 'loss/policy', 'llm_cache_hit_rate']
        for metric in key_metrics:
            values = [m.get(metric) for m in recent_metrics if metric in m]
            if values:
                trend = "↗" if values[-1] > values[0] else "↘" if values[-1] < values[0] else "→"
                dashboard.append(f"  {metric}: {values[-1]:.3f} {trend}")
    
    # Recommendations
    recommendations = monitor.get_recommendations()
    if recommendations:
        dashboard.append(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            dashboard.append(f"  {i}. {rec}")
    
    dashboard.append("\n" + "="*60)
    
    return "\n".join(dashboard)