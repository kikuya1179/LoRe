"""
PriorNet: Knowledge Distillation Network for LLM Prior
Reduces API calls by learning to mimic LLM policy guidance
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..conf import LoReConfig


class LLMDataPoint:
    """Single LLM response data point for distillation."""
    
    def __init__(self, obs: np.ndarray, context: Dict[str, Any], 
                 llm_logits: np.ndarray, llm_features: List[float] = None,
                 confidence: float = 1.0):
        self.obs = obs
        self.context = context  
        self.llm_logits = llm_logits
        self.llm_features = llm_features or []
        self.confidence = confidence
        self.timestamp = time.time()


class LLMDistillationDataset(Dataset):
    """Dataset for LLM response distillation."""
    
    def __init__(self, data_points: List[LLMDataPoint], max_age_hours: float = 24.0):
        self.data_points = data_points
        self.max_age_hours = max_age_hours
        self._filter_by_age()
    
    def _filter_by_age(self):
        """Remove data points older than max_age_hours."""
        current_time = time.time()
        cutoff_time = current_time - (self.max_age_hours * 3600)
        
        self.data_points = [
            dp for dp in self.data_points 
            if dp.timestamp >= cutoff_time
        ]
    
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dp = self.data_points[idx]
        
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(dp.obs).float()
        if obs_tensor.dim() == 3:  # Add batch dimension
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Convert context to feature vector (simplified for MiniGrid)
        context_features = self._encode_context(dp.context)
        
        # Target logits and confidence
        target_logits = torch.from_numpy(dp.llm_logits).float()
        confidence = torch.tensor(dp.confidence, dtype=torch.float32)
        
        return {
            'obs': obs_tensor,
            'context_features': context_features,
            'target_logits': target_logits,
            'confidence': confidence,
            'llm_features': torch.tensor(dp.llm_features, dtype=torch.float32) if dp.llm_features else torch.zeros(4)
        }
    
    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode MiniGrid context into feature vector."""
        mission = context.get('mission', '')
        remaining_steps = context.get('remaining_steps', 0)
        
        # Simple context encoding
        mission_hash = hash(mission) % 1000 / 1000.0  # Normalize to [0,1]
        step_ratio = min(remaining_steps / 100.0, 1.0)  # Normalize steps
        
        return torch.tensor([mission_hash, step_ratio], dtype=torch.float32)


class PriorNet(nn.Module):
    """Neural network to mimic LLM policy guidance."""
    
    def __init__(self, obs_channels: int = 1, latent_dim: int = 256, 
                 n_actions: int = 7, context_dim: int = 2):
        super().__init__()
        self.obs_channels = obs_channels
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.context_dim = context_dim
        
        # Observation encoder (similar to Dreamer but smaller)
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(latent_dim // 2)
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim // 4)
        )
        
        # Combined feature processor
        combined_dim = latent_dim // 2 + latent_dim // 4
        self.feature_processor = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU()
        )
        
        # Policy head (outputs logits)
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_actions)
        )
        
        # Feature head (optional, for LLM features prediction)
        self.feature_head = nn.Sequential(
            nn.Linear(latent_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 strategic features
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, obs: torch.Tensor, context_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through PriorNet."""
        # Encode observation
        obs_features = self.obs_encoder(obs)
        
        # Encode context
        context_enc = self.context_encoder(context_features)
        
        # Combine features
        combined = torch.cat([obs_features, context_enc], dim=-1)
        processed = self.feature_processor(combined)
        
        # Generate outputs
        logits = self.policy_head(processed)
        features = self.feature_head(processed)
        confidence = self.confidence_head(processed).squeeze(-1)
        
        return {
            'logits': logits,
            'features': features,
            'confidence': confidence
        }


class PriorNetDistiller:
    """Handles PriorNet training and distillation from LLM data."""
    
    def __init__(self, config: LoReConfig, obs_channels: int = 1, n_actions: int = 7):
        self.config = config
        self.n_actions = n_actions
        
        # PriorNet model
        self.priornet = PriorNet(obs_channels=obs_channels, n_actions=n_actions)
        self.optimizer = torch.optim.Adam(self.priornet.parameters(), lr=3e-4)
        
        # Data collection
        self.llm_data: deque = deque(maxlen=10000)  # Keep last 10k samples
        self.distillation_threshold = 2000  # Start distillation after 2k samples
        self.distillation_interval = 5000  # Distill every 5k samples collected
        self.samples_since_distillation = 0
        
        # Training state
        self.device = torch.device('cpu')
        self.training_epochs = 5
        self.batch_size = 64
        self.last_distillation_loss = float('inf')
        
        # Usage tracking
        self.total_predictions = 0
        self.api_calls_saved = 0
        self.performance_threshold = 0.8  # Switch to PriorNet when confidence > 0.8
        
    def to(self, device: torch.device):
        """Move PriorNet to device."""
        self.device = device
        self.priornet.to(device)
        return self
    
    def add_llm_sample(self, obs: np.ndarray, context: Dict[str, Any], 
                      llm_logits: np.ndarray, llm_features: List[float] = None,
                      confidence: float = 1.0):
        """Add LLM response sample for distillation."""
        data_point = LLMDataPoint(obs, context, llm_logits, llm_features, confidence)
        self.llm_data.append(data_point)
        self.samples_since_distillation += 1
        
        # Trigger distillation if enough samples collected
        if (len(self.llm_data) >= self.distillation_threshold and 
            self.samples_since_distillation >= self.distillation_interval):
            self._perform_distillation()
    
    def predict(self, obs: np.ndarray, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Predict using PriorNet instead of LLM."""
        if len(self.llm_data) < self.distillation_threshold:
            return None  # Not enough training data yet
        
        self.priornet.eval()
        with torch.no_grad():
            # Prepare inputs
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Encode context (reuse dataset method)
            dataset = LLMDistillationDataset([])
            context_features = dataset._encode_context(context).unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.priornet(obs_tensor, context_features)
            
            # Check confidence threshold
            confidence = float(outputs['confidence'])
            if confidence < self.performance_threshold:
                return None  # Low confidence, use LLM instead
            
            # Extract results
            logits = outputs['logits'].cpu().numpy().squeeze()
            features = outputs['features'].cpu().numpy().squeeze().tolist()
            
            self.total_predictions += 1
            self.api_calls_saved += 1
            
            return {
                'policy': {'logits': logits.tolist()},
                'features': features,
                'confidence': confidence,
                'mask': [1] * self.n_actions,  # All actions valid for MiniGrid
                'notes': f'PriorNet prediction (conf={confidence:.3f})'
            }
    
    def _perform_distillation(self):
        """Perform knowledge distillation from collected LLM data."""
        print(f"[PriorNet] Starting distillation with {len(self.llm_data)} samples...")
        
        # Create dataset and dataloader
        dataset = LLMDistillationDataset(list(self.llm_data))
        if len(dataset) < 32:
            print("[PriorNet] Not enough valid samples for distillation")
            return
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                              num_workers=0, drop_last=False)
        
        # Training loop
        self.priornet.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.training_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataloader:
                # Move to device
                obs = batch['obs'].to(self.device)
                context_features = batch['context_features'].to(self.device)
                target_logits = batch['target_logits'].to(self.device)
                confidence_weights = batch['confidence'].to(self.device)
                
                # Forward pass
                outputs = self.priornet(obs, context_features)
                
                # Compute losses
                # 1. Policy distillation loss (KL divergence)
                pred_logprobs = F.log_softmax(outputs['logits'], dim=-1)
                target_probs = F.softmax(target_logits, dim=-1)
                kl_loss = F.kl_div(pred_logprobs, target_probs, reduction='none').sum(dim=-1)
                
                # 2. Feature matching loss (if available)
                feature_loss = torch.tensor(0.0, device=self.device)
                if 'llm_features' in batch and batch['llm_features'].numel() > 0:
                    target_features = batch['llm_features'].to(self.device)
                    feature_loss = F.mse_loss(outputs['features'], target_features, reduction='none').mean(dim=-1)
                
                # 3. Confidence calibration loss
                confidence_loss = F.mse_loss(outputs['confidence'], confidence_weights, reduction='none')
                
                # Weighted combination
                total_batch_loss = (
                    kl_loss.mean() + 
                    0.1 * feature_loss.mean() + 
                    0.05 * confidence_loss.mean()
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.priornet.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += total_batch_loss.item()
                epoch_batches += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            total_loss += avg_epoch_loss
            num_batches += 1
            
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print(f"[PriorNet] Epoch {epoch+1}/{self.training_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Update state
        self.last_distillation_loss = total_loss / max(num_batches, 1)
        self.samples_since_distillation = 0
        
        print(f"[PriorNet] Distillation completed. Avg loss: {self.last_distillation_loss:.4f}")
        print(f"[PriorNet] API calls saved so far: {self.api_calls_saved}")
    
    def should_use_priornet(self) -> bool:
        """Decide whether to use PriorNet or LLM."""
        if len(self.llm_data) < self.distillation_threshold:
            return False
        
        # Use PriorNet if distillation loss is reasonable
        return self.last_distillation_loss < 2.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get PriorNet usage statistics."""
        api_calls_total = self.total_predictions + (len(self.llm_data) if self.llm_data else 0)
        api_savings_rate = self.api_calls_saved / max(api_calls_total, 1)
        
        return {
            'priornet_samples_collected': len(self.llm_data),
            'priornet_predictions': self.total_predictions,
            'priornet_api_calls_saved': self.api_calls_saved,
            'priornet_api_savings_rate': api_savings_rate,
            'priornet_last_distillation_loss': self.last_distillation_loss,
            'priornet_samples_since_distillation': self.samples_since_distillation,
            'priornet_ready': len(self.llm_data) >= self.distillation_threshold
        }
    
    def save(self, path: str):
        """Save PriorNet model."""
        torch.save({
            'priornet_state_dict': self.priornet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'last_distillation_loss': self.last_distillation_loss,
            'total_predictions': self.total_predictions,
            'api_calls_saved': self.api_calls_saved
        }, path)
    
    def load(self, path: str):
        """Load PriorNet model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.priornet.load_state_dict(checkpoint['priornet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.last_distillation_loss = checkpoint.get('last_distillation_loss', float('inf'))
        self.total_predictions = checkpoint.get('total_predictions', 0)
        self.api_calls_saved = checkpoint.get('api_calls_saved', 0)