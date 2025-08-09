from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMAdapterConfig:
    enabled: bool = False
    # API モード既定モデル（無料枠で高スループット）
    model: str = "gemini-2.5-flash-lite"
    timeout_s: float = 2.5
    features_dim: int = 0  # 0 means no features
    # CLI 経由で呼ぶ場合の設定（Python SDK 非依存）
    use_cli: bool = True
    cli_exe: str = "gemini"  # 例: gemini / gemini-cli


class LLMAdapter:
    """Thin wrapper around Gemini 2.5 (google-generativeai) with safe fallbacks.

    If disabled or SDK/API key is not available, returns zero priors and mask=0.
    This keeps training deterministic and cheap unless explicitly enabled.
    """

    def __init__(self, cfg: Optional[LLMAdapterConfig] = None) -> None:
        self.cfg = cfg or LLMAdapterConfig()
        self._client = None
        # CLI 優先。API はフォールバックとして infer() 内で遅延初期化。
        if self.cfg.enabled and not self.cfg.use_cli:
            try:
                import os
                import google.generativeai as genai  # type: ignore

                api_key = os.environ.get("GEMINI_API_KEY")
                # 任意: 環境変数 GEMINI_MODEL があれば上書き
                env_model = os.environ.get("GEMINI_MODEL")
                if env_model:
                    self.cfg.model = env_model
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

        if not self.cfg.enabled:
            return out

        # 1) CLI 優先
        if self.cfg.use_cli:
            try:
                text = self._infer_via_cli(num_actions=num_actions)
                import json
                data = json.loads(text)
                logits = np.array(
                    data.get("action_prior", {}).get("logits", []), dtype=np.float32
                )
                if logits.shape == (num_actions,):
                    conf = float(data.get("confidence", 0.0))
                    conf = np.array([np.clip(conf, 0.0, 1.0)], dtype=np.float32)
                    feats = np.array(data.get("features", []), dtype=np.float32)
                    if self.cfg.features_dim > 0:
                        if feats.shape != (self.cfg.features_dim,):
                            feats = np.zeros((self.cfg.features_dim,), dtype=np.float32)
                    else:
                        feats = np.zeros((0,), dtype=np.float32)
                    return {
                        "prior_logits": logits,
                        "confidence": conf,
                        "mask": np.ones((1,), dtype=np.float32),
                        "features": feats,
                    }
            except Exception:
                # 続いて API にフォールバック
                pass

        # 2) API フォールバック（遅延初期化）
        if self._ensure_api_client():
            try:
                import json
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FUTimeout
                prompt = self._build_prompt(num_actions)
                def _call():
                    return self._client.generate_content(prompt)  # type: ignore[attr-defined]
                with ThreadPoolExecutor(max_workers=1) as ex:
                    resp = ex.submit(_call).result(timeout=self.cfg.timeout_s)
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
                return {
                    "prior_logits": logits,
                    "confidence": conf,
                    "mask": np.ones((1,), dtype=np.float32),
                    "features": feats,
                }
            except (FUTimeout, Exception):
                return out
        # 3) 全て失敗 → デフォルト出力
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

    def _build_prompt(self, num_actions: int) -> str:
        return (
            "Given an observation from the Crafter environment, output a JSON with "
            "keys: action_prior.logits (array length A), confidence (0..1), and optionally features (array length K). "
            f"Here A={num_actions} and K={self.cfg.features_dim}. Keep it strictly JSON."
        )

    def _infer_via_cli(self, num_actions: int) -> str:
        """Gemini CLI を単発で呼び出し、標準出力のテキストを返す。

        - モデル指定は行わない（既定モデルを利用）
        - まず `gemini`（stdin パイプ）を試し、失敗時は `gemini-cli prompt` を試す
        """
        import subprocess

        prompt = self._build_prompt(num_actions)
        timeout = max(0.5, float(self.cfg.timeout_s))

        # 1) `gemini` に stdin で渡す
        try:
            proc = subprocess.run(
                [self.cfg.cli_exe],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout.strip()
        except Exception:
            pass

        # 2) `gemini-cli prompt "..."` を試す
        try:
            proc = subprocess.run(
                [self.cfg.cli_exe, "prompt", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout.strip()
        except Exception:
            pass

        raise RuntimeError("Gemini CLI invocation failed")

    def _ensure_api_client(self) -> bool:
        """必要に応じて API クライアントを初期化。利用可能なら True を返す。"""
        if self._client is not None:
            return True
        try:
            import os
            import google.generativeai as genai  # type: ignore
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return False
            env_model = os.environ.get("GEMINI_MODEL")
            if env_model:
                self.cfg.model = env_model
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.cfg.model)
            return True
        except Exception:
            return False


