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
    use_cli: bool = False
    cli_exe: str = "gemini"  # 例: gemini / gemini-cli
    api_retries: int = 2


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
                # no API key -> keep enabled True, lazy init will try later
            except Exception:
                # SDK not installed or init failed -> keep enabled, fallback at call time
                pass

    def infer(
        self,
        obs_np: Any,
        num_actions: int,
        context: Optional[Dict[str, Any]] = None,
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

        # Debug flag (controlled by env var to avoid noise)
        try:
            import os as _os
            _dbg_on = _os.environ.get("LORE_DEBUG_LLM", "0") in ("1", "true", "True")
        except Exception:
            _dbg_on = False

        if not self.cfg.enabled:
            if _dbg_on:
                print("[LLMAdapter] disabled -> returning zeros")
            return out

        # 1) CLI 優先
        if self.cfg.use_cli:
            if _dbg_on:
                print(f"[LLMAdapter] using CLI model={self.cfg.model} timeout={self.cfg.timeout_s}s num_actions={num_actions}")
            try:
                # NOTE: CLI経路は context を埋め込めないため、簡易プロンプトのみ
                text = self._infer_via_cli(num_actions=num_actions)
                import json
                data = json.loads(text)
                norm = self._normalize_response_data(data, num_actions)
                if norm is not None:
                    if _dbg_on:
                        print("[LLMAdapter] CLI OK -> prior_logits,len=", len(norm.get("prior_logits", [])))
                    return norm
            except Exception:
                # 続いて API にフォールバック
                if _dbg_on:
                    print("[LLMAdapter] CLI failed -> fallback API")
                pass

        # 2) API フォールバック（遅延初期化）
        if self._ensure_api_client():
            import json
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FUTimeout
            prompt = self._build_prompt(num_actions, context)
            if _dbg_on:
                ph = (prompt or "")[:240].replace("\n", " ")
                print(f"[LLMAdapter] using API model={self.cfg.model} timeout={self.cfg.timeout_s}s num_actions={num_actions} prompt_head='{ph}'")
            def _call():
                # Force JSON output to ease parsing
                try:
                    from google.generativeai.types import GenerationConfig  # type: ignore
                    gen_cfg = GenerationConfig(response_mime_type="application/json")
                    return self._client.generate_content(prompt, generation_config=gen_cfg)  # type: ignore[attr-defined]
                except Exception:
                    # Fallback without explicit generation_config
                    return self._client.generate_content(prompt)  # type: ignore[attr-defined]
            last_err = None
            # configurable retries via env or cfg (default 2)
            try:
                _retries = int(getattr(self.cfg, 'api_retries', 0)) or int(_os.environ.get('LORE_LLM_RETRIES', '2'))
            except Exception:
                _retries = 2
            for attempt in range(_retries):
                try:
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        resp = ex.submit(_call).result(timeout=self.cfg.timeout_s)
                    text = getattr(resp, "text", None) or "{}"
                    data = json.loads(text)
                    norm = self._normalize_response_data(data, num_actions)
                    if norm is not None:
                        if _dbg_on:
                            print("[LLMAdapter] API OK -> logits_shape=", len(norm.get("prior_logits", [])))
                        return norm
                except (FUTimeout, Exception) as e:
                    last_err = e
                    if _dbg_on:
                        msg = str(e)
                        print(f"[LLMAdapter] API ERROR (attempt {attempt+1}/2): {msg[:200]}")
                    # brief backoff
                    try:
                        import time as _t
                        _t.sleep(0.2)
                    except Exception:
                        pass
            # all attempts failed
            return out
        # 3) 全て失敗 → デフォルト出力
        if _dbg_on:
            print("[LLMAdapter] fallback zeros")
        return out

    def _build_prompt(self, num_actions: int, context: Optional[Dict[str, Any]] = None) -> str:
        base = (
            "You act as an action prior advisor for a discrete policy. "
            "Return strictly JSON. Keys: action_prior.logits (length A), confidence (0..1), optional features (length K). "
            f"Here A={num_actions} and K={self.cfg.features_dim}.\n"
        )
        if context and isinstance(context, dict):
            try:
                import json as _json
                ctx_text = _json.dumps(context)[:4000]
                base += "Context: " + ctx_text + "\n"
            except Exception:
                pass
        base += "Output JSON only, no markdown."
        return base

    def _normalize_response_data(self, data: Dict[str, Any], num_actions: int) -> Optional[Dict[str, Any]]:
        """Normalize various JSON shapes returned by providers into adapter output.

        Accepts:
        - {"action_prior": {"logits": [...]}, "confidence": x, "features": [...]} (preferred)
        - {"action_prior_logits": [...], "confidence": x}
        - {"action_prior.logits": [...], ...}
        - {"logits": [...], ...}
        """
        import numpy as _np

        logits_raw = None
        if isinstance(data.get("action_prior"), dict):
            logits_raw = data["action_prior"].get("logits") or data["action_prior"].get("prior")
        if logits_raw is None:
            logits_raw = data.get("action_prior_logits") or data.get("action_prior.logits") or data.get("logits")

        try:
            logits = _np.array(logits_raw, dtype=_np.float32)
        except Exception:
            logits = _np.zeros((0,), dtype=_np.float32)

        if logits.shape != (num_actions,):
            # Debug print but avoid spamming callers; keep it concise
            try:
                import os as _os
                if _os.environ.get("LORE_DEBUG_LLM", "0") in ("1", "true", "True"):
                    print(f"[LLMAdapter] shape mismatch logits_shape={getattr(logits, 'shape', None)} expected=({num_actions},)")
            except Exception:
                pass
            return None

        conf = data.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        conf_arr = _np.array([_np.clip(conf, 0.0, 1.0)], dtype=_np.float32)

        feats = data.get("features", [])
        feats_arr = _np.array(feats, dtype=_np.float32)
        if self.cfg.features_dim > 0:
            if feats_arr.shape != (self.cfg.features_dim,):
                feats_arr = _np.zeros((self.cfg.features_dim,), dtype=_np.float32)
        else:
            feats_arr = _np.zeros((0,), dtype=_np.float32)

        return {
            "prior_logits": logits,
            "confidence": conf_arr,
            "mask": _np.ones((1,), dtype=_np.float32),
            "features": feats_arr,
        }

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


