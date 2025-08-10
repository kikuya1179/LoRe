"""Knowledge distillation utilities for reducing LLM dependency."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for policy learning."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5, 
                 kl_weight: float = 1.0, mse_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Balance between distillation and task loss
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                student_values: torch.Tensor, teacher_values: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute distillation loss between student and teacher."""
        batch_size = student_logits.size(0)
        
        # Apply temperature scaling
        student_soft = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss (main distillation loss)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            teacher_soft,
            reduction='none'
        ).sum(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            kl_loss = kl_loss * mask
            
        kl_loss = kl_loss.mean() * (self.temperature ** 2)
        
        # Value function distillation (if teacher values provided)
        value_loss = torch.tensor(0.0, device=student_logits.device)
        if teacher_values is not None:
            value_loss = F.mse_loss(student_values, teacher_values)
        
        # Task loss (if targets provided)
        task_loss = torch.tensor(0.0, device=student_logits.device)
        if targets is not None:
            task_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = (self.alpha * kl_loss * self.kl_weight + 
                     (1 - self.alpha) * task_loss + 
                     value_loss * self.mse_weight)
        
        return {
            'total_loss': total_loss,
            'kl_loss': kl_loss,
            'value_loss': value_loss, 
            'task_loss': task_loss,
        }


class ProgressiveDistillation:
    """Progressive distillation to reduce LLM dependency over time."""
    
    def __init__(self, initial_alpha: float = 0.3, final_alpha: float = 0.01,
                 distill_every: int = 1000, total_steps: int = 100000):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.distill_every = distill_every
        self.total_steps = total_steps
        self.current_step = 0
        
        # Distillation dataset
        self.distill_buffer: List[Dict[str, torch.Tensor]] = []
        self.max_buffer_size = 10000
        
        # Performance tracking
        self.student_performance_history: List[float] = []
        self.teacher_performance_history: List[float] = []
        self.distillation_gap: List[float] = []
    
    def get_current_alpha(self) -> float:
        """Get current alpha value based on schedule."""
        progress = min(1.0, self.current_step / self.total_steps)
        # Exponential decay
        alpha = self.initial_alpha * (self.final_alpha / self.initial_alpha) ** progress
        return max(self.final_alpha, alpha)
    
    def should_distill(self) -> bool:
        """Check if distillation should be performed."""
        return self.current_step % self.distill_every == 0
    
    def collect_teacher_data(self, states: torch.Tensor, teacher_logits: torch.Tensor,
                           teacher_values: Optional[torch.Tensor] = None,
                           llm_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Collect teacher demonstrations for distillation."""
        batch_size = states.size(0)
        
        for i in range(batch_size):
            sample = {
                'state': states[i].cpu(),
                'teacher_logits': teacher_logits[i].cpu(),
            }
            
            if teacher_values is not None:
                sample['teacher_values'] = teacher_values[i].cpu()
            
            if llm_metadata:
                sample['metadata'] = {k: v[i] if torch.is_tensor(v) else v 
                                    for k, v in llm_metadata.items()}
            
            self.distill_buffer.append(sample)
        
        # Maintain buffer size
        if len(self.distill_buffer) > self.max_buffer_size:
            # Remove oldest samples (FIFO)
            excess = len(self.distill_buffer) - self.max_buffer_size
            self.distill_buffer = self.distill_buffer[excess:]
    
    def get_distillation_batch(self, batch_size: int = 128) -> Optional[Dict[str, torch.Tensor]]:
        """Sample batch for distillation training."""
        if len(self.distill_buffer) < batch_size:
            return None
        
        # Random sampling
        indices = np.random.choice(len(self.distill_buffer), batch_size, replace=False)
        samples = [self.distill_buffer[i] for i in indices]
        
        # Stack into batch
        batch = {}
        for key in samples[0].keys():
            if key != 'metadata':  # Skip metadata for batching
                batch[key] = torch.stack([s[key] for s in samples])
        
        return batch
    
    def update_performance(self, student_reward: float, teacher_reward: float) -> None:
        """Update performance tracking."""
        self.student_performance_history.append(student_reward)
        self.teacher_performance_history.append(teacher_reward)
        
        # Maintain history size
        max_history = 1000
        if len(self.student_performance_history) > max_history:
            self.student_performance_history.pop(0)
            self.teacher_performance_history.pop(0)
        
        # Compute distillation gap
        if len(self.student_performance_history) >= 100:
            recent_student = np.mean(self.student_performance_history[-100:])
            recent_teacher = np.mean(self.teacher_performance_history[-100:])
            gap = abs(recent_teacher - recent_student)
            self.distillation_gap.append(gap)
    
    def get_distillation_quality(self) -> Dict[str, float]:
        """Get metrics on distillation quality."""
        if not self.distillation_gap:
            return {'gap': 0.0, 'student_performance': 0.0, 'teacher_performance': 0.0}
        
        return {
            'gap': float(np.mean(self.distillation_gap[-100:])),
            'student_performance': float(np.mean(self.student_performance_history[-100:])),
            'teacher_performance': float(np.mean(self.teacher_performance_history[-100:])),
            'buffer_size': len(self.distill_buffer),
            'current_alpha': self.get_current_alpha(),
        }
    
    def step(self) -> None:
        """Advance distillation step."""
        self.current_step += 1
    
    def can_run_autonomous(self, threshold: float = 0.05) -> bool:
        """Check if student can run autonomously (gap < threshold)."""
        if len(self.distillation_gap) < 50:
            return False
        
        recent_gap = np.mean(self.distillation_gap[-50:])
        return recent_gap < threshold


class OnlineDistillation:
    """Online distillation during training."""
    
    def __init__(self, distill_loss: DistillationLoss, 
                 progressive_distill: ProgressiveDistillation):
        self.distill_loss = distill_loss
        self.progressive_distill = progressive_distill
        self.enabled = True
    
    def compute_distillation_loss(self, student_logits: torch.Tensor, 
                                teacher_logits: torch.Tensor,
                                student_values: torch.Tensor,
                                teacher_values: Optional[torch.Tensor] = None,
                                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute online distillation loss."""
        if not self.enabled:
            return {'total_loss': torch.tensor(0.0, device=student_logits.device)}
        
        # Adjust alpha based on progressive schedule
        current_alpha = self.progressive_distill.get_current_alpha()
        original_alpha = self.distill_loss.alpha
        self.distill_loss.alpha = current_alpha
        
        # Compute distillation loss
        loss_dict = self.distill_loss(
            student_logits, teacher_logits, student_values, 
            teacher_values, mask=mask
        )
        
        # Restore original alpha
        self.distill_loss.alpha = original_alpha
        
        return loss_dict
    
    def should_use_teacher(self) -> bool:
        """Decide whether to use teacher (LLM) for current step."""
        current_alpha = self.progressive_distill.get_current_alpha()
        
        # Use teacher with probability equal to current alpha
        return np.random.random() < current_alpha
    
    def update(self, student_reward: float, teacher_reward: Optional[float] = None) -> None:
        """Update distillation state."""
        self.progressive_distill.step()
        
        if teacher_reward is not None:
            self.progressive_distill.update_performance(student_reward, teacher_reward)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get distillation metrics."""
        quality_metrics = self.progressive_distill.get_distillation_quality()
        
        return {
            **quality_metrics,
            'distillation_enabled': self.enabled,
            'can_run_autonomous': self.progressive_distill.can_run_autonomous(),
            'should_use_teacher': self.should_use_teacher(),
            'distill_step': self.progressive_distill.current_step,
        }