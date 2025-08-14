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
    enabled: bool = True
    model: str = "gemini-2.5-flash-lite"
    timeout_s: float = 2.5
    api_retries: int = 2
    features_dim: int = 0
    use_cli: bool = False
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
    
    def __init__(self, num_actions: int, cfg: Optional[LLMAdapterConfigV2] = None):
        self.cfg = cfg or LLMAdapterConfigV2()
        self.num_actions = num_actions
        self._client = None
        self.code_store = CodeKVStore()
        self.dsl_executor = DSLExecutor(timeout_ms=int(self.cfg.timeout_s * 1000))
        
        # Temperature estimation for logits normalization
        self.temperature = 1.0
        self.temperature_estimated = False
        
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
    
    def get_action_logits(self, obs_np: np.ndarray, context: Dict[str, Any]) -> Optional[CodeOut]:
        """Get action logits for MiniGrid environment"""
        result = self.infer(obs_np, context, self.num_actions)
        if result and result.policy and 'logits' in result.policy:
            # Apply temperature scaling and centering
            logits = np.array(result.policy['logits'])
            if len(logits) == self.num_actions:
                # Center logits
                logits = logits - np.mean(logits)
                # Apply temperature scaling
                logits = logits / self.temperature
                result.policy['logits'] = logits.tolist()
                return result
        return None
    
    def estimate_temperature(self, sample_obs: List[np.ndarray], 
                           sample_contexts: List[Dict[str, Any]],
                           reference_logits: List[np.ndarray]) -> None:
        """Estimate temperature for logits alignment"""
        if self.temperature_estimated or not self.cfg.enabled:
            return
            
        llm_responses = self.infer_batch(sample_obs, sample_contexts, self.num_actions)
        valid_pairs = []
        
        for llm_resp, ref_logits in zip(llm_responses, reference_logits):
            if (llm_resp and llm_resp.policy and 'logits' in llm_resp.policy and 
                len(llm_resp.policy['logits']) == self.num_actions):
                llm_logits = np.array(llm_resp.policy['logits'])
                # Center both
                llm_logits = llm_logits - np.mean(llm_logits)
                ref_logits_centered = ref_logits - np.mean(ref_logits)
                valid_pairs.append((llm_logits, ref_logits_centered))
        
        if len(valid_pairs) >= 3:
            # Find temperature that minimizes softmax distance
            best_temp = 1.0
            best_dist = float('inf')
            
            for temp in np.linspace(0.5, 3.0, 20):
                total_dist = 0.0
                for llm_logits, ref_logits in valid_pairs:
                    scaled_llm = llm_logits / temp
                    llm_probs = np.exp(scaled_llm) / np.sum(np.exp(scaled_llm))
                    ref_probs = np.exp(ref_logits) / np.sum(np.exp(ref_logits))
                    total_dist += np.sum((llm_probs - ref_probs) ** 2)
                
                if total_dist < best_dist:
                    best_dist = total_dist
                    best_temp = temp
            
            self.temperature = best_temp
            self.temperature_estimated = True
    
    def _build_prompt(self, obs: np.ndarray, context: Dict[str, Any], 
                     num_actions: int) -> str:
        """Build LLM prompt for MiniGrid environment."""
        # Extract MiniGrid specific context
        mission = context.get('mission', '')
        remaining_steps = context.get('remaining_steps', 0)
        
        # Simplified observation representation for MiniGrid
        # Convert grid to symbolic representation
        obs_str = self._grid_to_string(obs)
        
        # Field of view representation
        fov_summary = {
            'size': list(obs.shape[:2]),
            'agent_view': 'partial' if obs.shape[0] < 20 else 'full'
        }
        
        # MiniGrid specific prompt
        prompt = f"""You are a MiniGrid navigation agent. Analyze the current state and provide action guidance.

Grid View:
{obs_str}

Mission: {mission}
Remaining Steps: {remaining_steps}
Actions: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done (total: {num_actions})

Return JSON with action logits and analysis:
{{
    "policy": {{"logits": [{num_actions} float values]}},
    "mask": [1, 1, 1, 1, 1, 1, 1],
    "confidence": 0.0-1.0,
    "features": [2-5 strategic features in range [-2, 2]],
    "r_shaped": small reward bonus/penalty [-0.1, 0.1],
    "notes": "brief reasoning"
}}

Focus on:
- Navigation toward goal objects
- Avoiding obstacles/walls
- Efficient pathfinding
- Mission completion strategy

Example: {{"policy": {{"logits": [0.1, -0.2, 0.8, -0.5, -0.3, 0.0, -0.1]}}, "mask": [1]*{num_actions}, "confidence": 0.7}}"""
        
        return prompt
    
    def _grid_to_string(self, obs: np.ndarray) -> str:
        """Convert MiniGrid observation to text representation"""
        if len(obs.shape) != 3 or obs.shape[2] != 3:
            return "Invalid observation format"
        
        # MiniGrid object encoding (simplified)
        obj_map = {
            0: ' ',  # empty
            1: '#',  # wall  
            2: 'D',  # door
            3: 'K',  # key
            4: 'B',  # ball
            5: 'X',  # box
            6: 'G',  # goal
            10: '@', # agent
        }
        
        grid_str = ""
        height, width = obs.shape[:2]
        
        for y in range(height):
            row = ""
            for x in range(width):
                # Get object type from first channel
                obj_type = int(obs[y, x, 0])
                char = obj_map.get(obj_type, '?')
                row += char
            grid_str += row + "\n"
        
        return grid_str.strip()
    
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
        """Call API in batch mode with JSON forcing and simple retries."""
        if not self._client:
            return [""] * len(prompts)

        def call_single(prompt: str) -> str:
            retries = max(1, int(getattr(self.cfg, 'api_retries', 2)))
            last_err = None
            for _ in range(retries):
                try:
                    # Try to force JSON
                    try:
                        from google.generativeai.types import GenerationConfig  # type: ignore
                        gen_cfg = GenerationConfig(response_mime_type="application/json")
                        response = self._client.generate_content(prompt, generation_config=gen_cfg)
                    except Exception:
                        response = self._client.generate_content(prompt)
                    return getattr(response, 'text', '') or ''
                except Exception as e:
                    last_err = e
                    try:
                        import time as _t
                        _t.sleep(0.1)
                    except Exception:
                        pass
            return ""

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(call_single, prompt) for prompt in prompts]
            results: List[str] = []
            for future in futures:
                try:
                    result = future.result(timeout=self.cfg.timeout_s)
                    results.append(result)
                except (FutureTimeout, Exception):
                    results.append("")
        return results
    
    def _process_response(self, response: str, obs: np.ndarray, 
                         context: Dict[str, Any], num_actions: int) -> Optional[CodeOut]:
        """Process LLM response into CodeOut with enhanced validation."""
        if not response.strip():
            return None
        
        try:
            # Parse JSON response
            data = json.loads(response)
            
            # Extract and validate logits
            policy_data = data.get('policy', {})
            logits = policy_data.get('logits', [])
            
            if not isinstance(logits, list) or len(logits) != num_actions:
                return None
            
            # Validate logits are numbers and not NaN/inf
            try:
                logits_array = np.array(logits, dtype=np.float32)
                if np.any(np.isnan(logits_array)) or np.any(np.isinf(logits_array)):
                    return None
                # Clamp extreme values
                logits_array = np.clip(logits_array, -10.0, 10.0)
                logits = logits_array.tolist()
            except (ValueError, TypeError):
                return None
            
            # Validate mask
            mask = data.get('mask', [1] * num_actions)
            if not isinstance(mask, list) or len(mask) != num_actions:
                mask = [1] * num_actions
            
            # Ensure mask contains only 0s and 1s
            mask = [1 if x else 0 for x in mask]
            
            # Validate other fields
            features = data.get('features', [])
            if not isinstance(features, list):
                features = []
            
            r_shaped = data.get('r_shaped', 0.0)
            try:
                r_shaped = float(r_shaped)
                r_shaped = np.clip(r_shaped, -0.2, 0.2)
            except (ValueError, TypeError):
                r_shaped = 0.0
            
            confidence = data.get('confidence', 0.5)
            try:
                confidence = float(confidence)
                confidence = np.clip(confidence, 0.0, 1.0)
            except (ValueError, TypeError):
                confidence = 0.5
            
            result = CodeOut(
                features=features,
                subgoal=data.get('subgoal'),
                r_shaped=r_shaped,
                policy={'logits': logits},
                mask=mask,
                confidence=confidence,
                notes=str(data.get('notes', '')),
            )
            
            if result.validate(num_actions):
                return result
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
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