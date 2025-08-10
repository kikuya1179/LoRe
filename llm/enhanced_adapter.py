"""Enhanced LLM adapter with batch processing, caching, and schema validation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..utils.code_out import CodeOut, CodeKVStore, compute_state_hash
from .dsl_executor import DSLExecutor


@dataclass
class LLMAdapterConfigV2:
    enabled: bool = False
    model: str = "gemini-2.5-flash-lite"
    timeout_s: float = 2.5
    features_dim: int = 0
    use_cli: bool = True
    cli_exe: str = "gemini"
    
    # Enhanced config
    batch_size: int = 8
    cache_size: int = 1000
    use_dsl: bool = True  # Use DSL instead of free-form code
    novelty_threshold: float = 0.1  # For triggering LLM calls
    td_error_threshold: float = 0.5
    plateau_frames: int = 10000  # Frames without improvement to trigger LLM


class EnhancedLLMAdapter:
    """Enhanced LLM adapter with batching, caching, and safe execution."""
    
    def __init__(self, cfg: Optional[LLMAdapterConfigV2] = None):
        self.cfg = cfg or LLMAdapterConfigV2()
        self._client = None
        self.code_store = CodeKVStore()
        self.dsl_executor = DSLExecutor(timeout_ms=int(self.cfg.timeout_s * 1000))
        
        # Caching
        self._cache: Dict[str, Tuple[CodeOut, float]] = {}  # hash -> (result, timestamp)
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Batch processing
        self._pending_batch: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
        
        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.avg_execution_time = 0.0
        
        # Triggers
        self.last_performance = 0.0
        self.plateau_counter = 0
        
        if self.cfg.enabled and not self.cfg.use_cli:
            self._init_api_client()
    
    def _init_api_client(self) -> None:
        """Initialize API client if available."""
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(self.cfg.model)
            else:
                self.cfg.enabled = False
        except Exception:
            self.cfg.enabled = False
    
    def should_call_llm(self, obs: np.ndarray, td_error: float = 0.0, 
                       performance: float = 0.0, novelty: float = 0.0) -> bool:
        """Determine if LLM should be called based on triggers."""
        if not self.cfg.enabled:
            return False
        
        # Novelty trigger
        if novelty > self.cfg.novelty_threshold:
            return True
        
        # TD error trigger  
        if td_error > self.cfg.td_error_threshold:
            return True
        
        # Plateau trigger
        if abs(performance - self.last_performance) < 0.01:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0
            self.last_performance = performance
        
        if self.plateau_counter > self.cfg.plateau_frames:
            self.plateau_counter = 0  # Reset
            return True
        
        return False
    
    def infer_batch(self, obs_batch: List[np.ndarray], 
                   contexts: List[Dict[str, Any]], 
                   num_actions: int) -> List[Optional[CodeOut]]:
        """Process batch of observations."""
        if not self.cfg.enabled:
            return [None] * len(obs_batch)
        
        results = []
        cache_keys = []
        
        # Check cache first
        for obs, context in zip(obs_batch, contexts):
            cache_key = self._compute_cache_key(obs, context)
            cache_keys.append(cache_key)
            
            if cache_key in self._cache:
                cached_result, timestamp = self._cache[cache_key]
                # Cache valid for 5 minutes
                if time.time() - timestamp < 300:
                    results.append(cached_result)
                    self._cache_hits += 1
                    continue
            
            results.append(None)
            self._cache_misses += 1
        
        # Batch process uncached items
        uncached_indices = [i for i, result in enumerate(results) if result is None]
        
        if uncached_indices:
            uncached_obs = [obs_batch[i] for i in uncached_indices]
            uncached_contexts = [contexts[i] for i in uncached_indices]
            
            batch_results = self._process_batch(uncached_obs, uncached_contexts, num_actions)
            
            # Store in cache and results
            for idx, result in zip(uncached_indices, batch_results):
                if result is not None:
                    self._cache[cache_keys[idx]] = (result, time.time())
                    self._clean_cache()  # Periodic cleanup
                results[idx] = result
        
        return results
    
    def infer(self, obs: np.ndarray, context: Dict[str, Any], 
             num_actions: int) -> Optional[CodeOut]:
        """Single observation inference (uses batch processing internally)."""
        results = self.infer_batch([obs], [context], num_actions)
        return results[0] if results else None
    
    def _compute_cache_key(self, obs: np.ndarray, context: Dict[str, Any]) -> str:
        """Compute cache key for observation and context."""
        state_hash = compute_state_hash(obs, context)
        return f"{self.cfg.model}_{state_hash}"
    
    def _process_batch(self, obs_batch: List[np.ndarray], 
                      contexts: List[Dict[str, Any]], 
                      num_actions: int) -> List[Optional[CodeOut]]:
        """Process batch of observations through LLM."""
        start_time = time.time()
        
        try:
            # Generate prompts
            prompts = [self._build_prompt(obs, context, num_actions) 
                      for obs, context in zip(obs_batch, contexts)]
            
            if self.cfg.use_cli:
                responses = self._call_cli_batch(prompts)
            else:
                responses = self._call_api_batch(prompts)
            
            # Process responses
            results = []
            for i, response in enumerate(responses):
                try:
                    result = self._process_response(response, obs_batch[i], 
                                                  contexts[i], num_actions)
                    results.append(result)
                    if result is not None:
                        self.successful_calls += 1
                    else:
                        self.failed_calls += 1
                except Exception:
                    results.append(None)
                    self.failed_calls += 1
            
            # Update metrics
            execution_time = time.time() - start_time
            self.total_calls += len(obs_batch)
            self.avg_execution_time = (self.avg_execution_time * 0.9 + 
                                     execution_time * 0.1)
            
            return results
            
        except Exception:
            self.failed_calls += len(obs_batch)
            return [None] * len(obs_batch)
    
    def _build_prompt(self, obs: np.ndarray, context: Dict[str, Any], 
                     num_actions: int) -> str:
        """Build LLM prompt for Crafter environment."""
        # Observation summary
        obs_summary = {
            'shape': list(obs.shape),
            'mean_value': float(np.mean(obs)),
            'unique_colors': len(np.unique(obs.flatten()[:100])),  # Sample
        }
        
        # Context summary
        inventory = context.get('inventory', {})
        health = context.get('health', 100)
        
        if self.cfg.use_dsl:
            prompt = f"""You are a Crafter game planner. Given the observation and context below, generate a DSL expression that returns a dictionary with game analysis.

Observation: {obs_summary}
Context: inventory={inventory}, health={health}
Actions: {num_actions} discrete actions available

Return ONLY a valid Python dictionary expression using these DSL functions:
- count_pixels(color): Count pixels of specific color
- has_item(item): Check if item in inventory  
- distance_to(target): Distance to target (0-1)
- goal_progress(goal): Progress toward goal (0-1)
- health_ratio(): Health as ratio (0-1)
- inventory_full(): Whether inventory is full

Example output:
{{
    "features": [count_pixels(64)/100.0, distance_to("wood"), health_ratio()-0.5],
    "subgoal": {{"id": "collect_wood", "tau": 10}} if not has_item("wood") else None,
    "r_shaped": 0.1 if has_item("wood") else -0.05,
    "policy": {{"logits": [0.2, -0.1] + [0.0] * 15}},
    "mask": [1] * {num_actions},
    "confidence": 0.8,
    "notes": "Collecting resources"
}}

Features must be in range [-3, 3], r_shaped in [-0.2, 0.2], confidence in [0, 1].
"""
        else:
            # Free-form JSON prompt
            prompt = f"""Analyze this Crafter game state and return strategic guidance as JSON.

Observation shape: {obs_summary['shape']}, mean: {obs_summary['mean_value']:.3f}
Inventory: {inventory}  
Health: {health}
Available actions: {num_actions}

Return JSON with:
- "features": list of 2-8 numeric features [-3, 3] describing game state
- "r_shaped": small reward shaping [-0.2, 0.2] for current situation  
- "policy": {{"logits": [{num_actions} values]}} for action preferences
- "mask": [{num_actions} values of 0 or 1] for valid actions
- "confidence": 0.0-1.0 confidence in recommendations
- "subgoal": {{"id": "goal_name", "tau": steps}} for multi-step objectives
- "notes": brief explanation

Example: {{"features": [0.5, -0.2], "r_shaped": 0.05, "confidence": 0.7, "mask": [1]*{num_actions}}}"""
        
        return prompt
    
    def _call_cli_batch(self, prompts: List[str]) -> List[str]:
        """Call CLI in batch mode."""
        import subprocess
        
        results = []
        for prompt in prompts:
            try:
                proc = subprocess.run(
                    [self.cfg.cli_exe],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_s,
                    shell=False,
                )
                if proc.returncode == 0 and proc.stdout:
                    results.append(proc.stdout.strip())
                else:
                    results.append("")
            except Exception:
                results.append("")
        
        return results
    
    def _call_api_batch(self, prompts: List[str]) -> List[str]:
        """Call API in batch mode."""
        if not self._client:
            return [""] * len(prompts)
        
        def call_single(prompt: str) -> str:
            try:
                response = self._client.generate_content(prompt)
                return getattr(response, 'text', '') or ''
            except Exception:
                return ""
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(call_single, prompt) for prompt in prompts]
            results = []
            
            for future in futures:
                try:
                    result = future.result(timeout=self.cfg.timeout_s)
                    results.append(result)
                except (FutureTimeout, Exception):
                    results.append("")
        
        return results
    
    def _process_response(self, response: str, obs: np.ndarray, 
                         context: Dict[str, Any], num_actions: int) -> Optional[CodeOut]:
        """Process LLM response into CodeOut."""
        if not response.strip():
            return None
        
        try:
            if self.cfg.use_dsl:
                # Execute DSL code safely
                result = self.dsl_executor.execute(response, obs, context)
                if result and result.validate(num_actions):
                    return result
            else:
                # Parse JSON response
                data = json.loads(response)
                result = CodeOut(
                    features=data.get('features', []),
                    subgoal=data.get('subgoal'),
                    r_shaped=float(data.get('r_shaped', 0.0)),
                    policy=data.get('policy'),
                    mask=data.get('mask'),
                    confidence=float(data.get('confidence', 0.5)),
                    notes=str(data.get('notes', '')),
                )
                
                if result.validate(num_actions):
                    return result
            
        except Exception:
            pass
        
        return None
    
    def _clean_cache(self) -> None:
        """Clean old cache entries."""
        if len(self._cache) <= self.cfg.cache_size:
            return
        
        current_time = time.time()
        # Remove entries older than 30 minutes
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > 1800
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        # If still too large, remove oldest entries
        if len(self._cache) > self.cfg.cache_size:
            sorted_items = sorted(
                self._cache.items(), 
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            keep_items = sorted_items[-self.cfg.cache_size//2:]  # Keep newest half
            self._cache = dict(keep_items)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_cache_requests 
                         if total_cache_requests > 0 else 0.0)
        
        success_rate = (self.successful_calls / max(1, self.total_calls))
        
        return {
            'total_calls': self.total_calls,
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._cache),
            'avg_execution_time': self.avg_execution_time,
            'plateau_counter': self.plateau_counter,
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        self.code_store.clear_cache()
        self._cache_hits = 0
        self._cache_misses = 0