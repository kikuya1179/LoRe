from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMAdapterConfig:
    enabled: bool = False
    model: str = "gemini-1.5-pro"
    timeout_s: float = 2.5
    features_dim: int = 0  # 0 means no features


class LLMAdapter:
    """Thin wrapper around Gemini 2.5 (google-generativeai) with safe fallbacks.

    If disabled or SDK/API key is not available, returns zero priors and mask=0.
    This keeps training deterministic and cheap unless explicitly enabled.
    """

    def __init__(self, cfg: Optional[LLMAdapterConfig] = None) -> None:
        self.cfg = cfg or LLMAdapterConfig()
        self._client = None
        if self.cfg.enabled:
            try:
                import os
                import google.generativeai as genai  # type: ignore

                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self._client = genai.GenerativeModel(self.cfg.model)
                else:
                    # no API key -> operate in disabled mode
                    self.cfg.enabled = False
            except Exception:
                # SDK not installed or init failed -> disabled mode
                self.cfg.enabled = False

    def infer(
        self,
        obs_np: Any,
        num_actions: int,
    ) -> Dict[str, Any]:
        """Return dict with keys: prior_logits [A], confidence [1], mask [1], features [K].

        - When disabled or failure: zeros and mask=0
        - obs_np can be a numpy array or nested structure; kept opaque here
        """
        import numpy as np

        # defaults (disabled/failure)
        out = {
            "prior_logits": np.zeros((num_actions,), dtype=np.float32),
            "confidence": np.zeros((1,), dtype=np.float32),
            "mask": np.zeros((1,), dtype=np.float32),
            "features": np.zeros((self.cfg.features_dim,), dtype=np.float32)
            if self.cfg.features_dim > 0
            else np.zeros((0,), dtype=np.float32),
        }

        if not self.cfg.enabled or self._client is None:
            return out

        try:
            import json
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FUTimeout

            prompt = (
                "Given an observation from the Crafter environment, output a JSON with "
                "keys: action_prior.logits (array length A), confidence (0..1), and optionally features (array length K). "
                f"Here A={num_actions} and K={self.cfg.features_dim}. Keep it strictly JSON."
            )

            def _call():
                return self._client.generate_content(prompt)  # type: ignore[attr-defined]

            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_call)
                resp = fut.result(timeout=self.cfg.timeout_s)

            text = getattr(resp, "text", None) or "{}"
            data = json.loads(text)

            logits = np.array(
                data.get("action_prior", {}).get("logits", []), dtype=np.float32
            )
            if logits.shape != (num_actions,):
                return out

            conf = float(data.get("confidence", 0.0))
            conf = np.array([np.clip(conf, 0.0, 1.0)], dtype=np.float32)

            feats = np.array(data.get("features", []), dtype=np.float32)
            if self.cfg.features_dim > 0:
                if feats.shape != (self.cfg.features_dim,):
                    feats = np.zeros((self.cfg.features_dim,), dtype=np.float32)
            else:
                feats = np.zeros((0,), dtype=np.float32)

            out = {
                "prior_logits": logits,
                "confidence": conf,
                "mask": np.ones((1,), dtype=np.float32),
                "features": feats,
            }
            return out
        except (FUTimeout, Exception):
            return out


