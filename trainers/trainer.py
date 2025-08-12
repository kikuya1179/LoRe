from __future__ import annotations

from dataclasses import asdict
from typing import Any
import time

import torch

from ..utils.replay import make_replay
try:
    # Optional GPU/Hybrid replay (falls back if unavailable)
    from ..utils.gpu_replay import create_optimized_replay  # type: ignore
except Exception:  # pragma: no cover
    create_optimized_replay = None  # type: ignore
from ..utils.llm_adapter import LLMAdapter, LLMAdapterConfig
from ..utils.rnd import RND
from ..utils.hw_trace import log_event, set_log_path as hw_set_log_path, set_current_step as hw_set_step


class Trainer:
    def __init__(self, env, agent, logger, cfg, device) -> None:
        self.env = env
        self.agent = agent
        self.logger = logger
        self.cfg = cfg
        self.device = device
        # process identity for prints
        try:
            import os as _os, platform as _pf
            self._proc = f"pid={_os.getpid()}@{_pf.node()}"
        except Exception:
            self._proc = "pid=?"
        self._t0 = time.time()
        
        # GPU monitoring setup
        if device.type == "cuda":
            print(f"[proc:{self._proc}] [GPU] Trainer initialized on {device}")
            self._gpu_monitoring = True
        else:
            self._gpu_monitoring = False

        # Replay buffer (prefer GPU/Hybrid if enabled and CUDA available)
        self._replay_is_gpu = False
        try:
            use_gpu_rb = (self.device.type == "cuda") and bool(getattr(self.cfg.train, "replay_use_gpu", False))
        except Exception:
            use_gpu_rb = False

        if use_gpu_rb and create_optimized_replay is not None:
            try:
                self.replay = create_optimized_replay(capacity=cfg.train.replay_capacity, device=self.device)
                self._replay_is_gpu = True
                if self._gpu_monitoring:
                    print(f"[proc:{self._proc}] [GPU] Using Hybrid/GPU replay buffer")
            except Exception:
                # Fallback to standard CPU replay
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.replay = make_replay(
                        capacity=cfg.train.replay_capacity,
                        use_priority=bool(getattr(cfg.train, "use_priority_replay", False)),
                    )
                self._replay_is_gpu = False
        else:
            # Use standard replay buffer
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.replay = make_replay(
                    capacity=cfg.train.replay_capacity,
                    use_priority=bool(getattr(cfg.train, "use_priority_replay", False)),
                )

        # Specs
        try:
            self.obs_key = "pixels" if "pixels" in env.observation_spec.keys(True, True) else "observation"
        except Exception:
            self.obs_key = "observation"

        # Reset environment (Gymnasium API via TorchRL wrapper)
        td = self.env.reset()
        # TorchRL GymWrapper returns TensorDict with key "observation" by default
        self._last_td = td

        self.global_step = 0  # vector steps (env batch進行回数)
        self.frames_seen = 0  # 合算フレーム数（全環境の合計）
        # Concurrency primitives
        try:
            import threading as _threading
            self._replay_lock = _threading.Lock()
        except Exception:
            self._replay_lock = None
        # HW trace file setup (default under log_dir)
        try:
            log_dir = getattr(self.cfg, 'log_dir', 'runs/dreamer_crafter')
            hw_set_log_path(f"{log_dir}/hw_trace.txt")
        except Exception:
            pass

        # Determine number of parallel environments (batch size)
        try:
            bs = td.batch_size
            self._n_envs = int(bs[0]) if len(bs) > 0 else 1
        except Exception:
            self._n_envs = 1

        # LLM adapter (Gemini 2.5 via google-generativeai)
        self.llm = LLMAdapter(
            LLMAdapterConfig(
                enabled=getattr(self.cfg.train, "use_llm", False),
                model=getattr(self.cfg.train, "llm_model", "gemini-2.5-flash-lite"),
                features_dim=getattr(self.cfg.model, "llm_features_dim", 0),
                use_cli=getattr(self.cfg.train, "llm_use_cli", False),
                timeout_s=getattr(self.cfg.train, "llm_timeout_s", 2.5),
                api_retries=int(getattr(self.cfg.train, "llm_api_retries", 2)),
            )
        )

        # LLM call throttling / cache state
        self._llm_calls_total = 0
        # Per-env cooldown and counters
        import torch as _torch  # local alias
        self._llm_steps_since_call_env = _torch.full((self._n_envs,), 10**9, dtype=_torch.long)
        self._llm_calls_env = _torch.zeros(self._n_envs, dtype=_torch.long)
        self._llm_call_budget = int(getattr(self.cfg.train, "llm_call_budget_total", 0) or 0)
        self._llm_cooldown = int(getattr(self.cfg.train, "llm_cooldown_steps", 0) or 0)
        self._llm_beta_thr = float(getattr(self.cfg.train, "llm_beta_call_threshold", 1.0))
        self._llm_use_cache = bool(getattr(self.cfg.train, "llm_use_cache", True))
        self._llm_cache = {}  # simple dict cache: key->(logits, features, conf)
        self._llm_cache_hits = 0
        self._llm_cache_misses = 0
        self._llm_errors_total = 0
        self._llm_intervals = []  # steps between valid calls
        self._llm_beta_when_call = []
        self._priornet_used_total = 0
        self._priornet_opportunities_total = 0

        # Boundary / progress trackers (per-env)
        self._just_reset = _torch.ones(self._n_envs, dtype=_torch.bool)
        self._no_reward_steps = _torch.zeros(self._n_envs, dtype=_torch.long)

        # PriorNet for distillation (optional)
        self._priornet = None
        self._priornet_opt = None
        self._priornet_last_update = 0
        if bool(getattr(self.cfg.train, "llm_priornet_enabled", True)):
            try:
                in_dim = int(getattr(self.cfg.model, "latent_dim", 256))
                hid = int(getattr(self.cfg.train, "llm_priornet_hidden", 256))
                n_act = int(getattr(self.agent, "n_actions", 0))
                self._priornet = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hid),
                    torch.nn.ELU(),
                    torch.nn.Linear(hid, n_act),
                ).to(self.device)
                self._priornet_opt = torch.optim.Adam(self._priornet.parameters(), lr=float(getattr(self.cfg.train, "llm_priornet_lr", 1e-3)))
            except Exception:
                self._priornet = None
                self._priornet_opt = None

        # 行動ヒストグラム用バケット
        self._action_hist = None

        # RND (intrinsic reward)
        self._rnd = None
        if getattr(self.cfg.train, "use_intrinsic", False):
            try:
                ch = int(getattr(self.cfg.model, "obs_channels", 4))
                self._rnd = RND(in_channels=ch).to(self.device)
            except Exception:
                self._rnd = None

        # Random exploration warmup
        self._random_warmup_steps = int(getattr(self.cfg.train, "init_random_frames", 0))
        self._log_interval = int(getattr(self.cfg.train, "log_interval", 1000))
        self._save_every = int(getattr(self.cfg.train, "save_every_frames", 0))
        # Logging cadence trackers
        try:
            self._log_interval = int(getattr(self.cfg.train, "log_interval", 1000))
        except Exception:
            self._log_interval = 1000
        self._next_log_step = self._log_interval
        self._last_logs: dict[str, float] = {}
        self._next_gpu_report = 500

        # Episode-level tracking for returns and success rate
        try:
            import torch as _torch
            self._ep_ret_env = _torch.zeros(self._n_envs, dtype=_torch.float32)
            self._ep_len_env = _torch.zeros(self._n_envs, dtype=_torch.long)
            self._ep_success_env = _torch.zeros(self._n_envs, dtype=_torch.bool)
        except Exception:
            self._ep_ret_env = None
            self._ep_len_env = None
            self._ep_success_env = None
        self._success_hist: list[int] = []  # 1 for success, 0 otherwise
        self._success_hist_cap = 100
        # Crafter achievements tracking
        self._ach_counts: dict[str, int] = {}
        self._ach_episodes: int = 0

    def _get_obs_tensor(self, td) -> torch.Tensor:
        # Root 'observation' か、'next' 配下から抽出
        obs = td.get(self.obs_key)
        if obs is None and ("next", self.obs_key) in td.keys(True, True):
            obs = td.get(("next", self.obs_key))
        # ensure channels-first [C,H,W]
        if obs.dim() == 3 and obs.shape[-1] in (1, 3) and obs.shape[0] not in (1, 3):
            obs = obs.permute(2, 0, 1).contiguous()
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        # dtype to float32; scale if uint8
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255.0)
        else:
            obs = obs.float()
        return obs.to(self.device, non_blocking=True)

    def _append_to_replay(self, prev_td, action, next_td, llm_out: dict | None):
        # Build tensordict samples; vectorized path prefers single batched extend() for速度
        try:
            from tensordict import TensorDict  # TorchRL 0.9+ は独立パッケージ提供
        except Exception:  # pragma: no cover
            from torchrl.data import TensorDict  # 古いTorchRL向けフォールバック

        # Extract possibly batched fields (avoid boolean tensor in `or`)
        def _get_with_fallback(td_in, key_primary, key_fallback=None):
            val = td_in.get(key_primary)
            if val is None and key_fallback is not None:
                val = td_in.get(key_fallback)
            return val

        prev_obs = _get_with_fallback(prev_td, self.obs_key, ("next", self.obs_key))
        next_obs = _get_with_fallback(next_td, ("next", self.obs_key), self.obs_key)
        rew = _get_with_fallback(next_td, ("next", "reward"), "reward")
        dn = _get_with_fallback(next_td, ("next", "done"), "done")

        def _ensure_tensor(x, default=0.0):
            if x is None:
                return torch.tensor(default)
            return x if isinstance(x, torch.Tensor) else torch.tensor(x)

        prev_obs = _ensure_tensor(prev_obs)
        next_obs = _ensure_tensor(next_obs)
        rew = _ensure_tensor(rew)
        done = _ensure_tensor(dn, 0.0)
        act = _ensure_tensor(action)

        # If batched on env dimension, split and add per-env
        def _env_batch_size(t: torch.Tensor, obs_like: bool = False) -> int:
            if t is None or not isinstance(t, torch.Tensor):
                return 1
            # observation [B,C,H,W] → env batch = B; action [B,1] → env batch = B
            if obs_like and t.ndim >= 4:
                return t.shape[0]
            if (not obs_like) and t.ndim >= 2:
                return t.shape[0]
            return 1

        n = max(
            _env_batch_size(prev_obs, obs_like=True),
            _env_batch_size(act, obs_like=False),
            _env_batch_size(rew, obs_like=False),
            _env_batch_size(done, obs_like=False),
        )
        # If GPU replay is enabled, store normalized float tensors on GPU to avoid HtoD later
        if getattr(self, "_replay_is_gpu", False):
            # Helper to ensure column tensors on the correct device/dtype
            def _col_gpu(x: torch.Tensor | float | int, n_envs: int, dtype: torch.dtype) -> torch.Tensor:
                t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
                t = t.to(device=self.device, dtype=dtype)
                if t.numel() == 1:
                    return t.view(1, 1).repeat(n_envs, 1)
                if t.ndim == 1:
                    return t.view(-1, 1)
                return t

            # Build observation tensors on GPU in float32 [0,1]
            try:
                o_prev_gpu = self._get_obs_tensor(prev_td)  # [N,C,H,W] or [1,C,H,W]
            except Exception:
                o_prev_gpu = torch.as_tensor(prev_obs, device=self.device, dtype=torch.float32)
                if o_prev_gpu.ndim == 3:
                    o_prev_gpu = o_prev_gpu.unsqueeze(0)
            try:
                o_next_gpu = self._get_obs_tensor(next_td)
            except Exception:
                o_next_gpu = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
                if o_next_gpu.ndim == 3:
                    o_next_gpu = o_next_gpu.unsqueeze(0)

            # Actions / rewards / done on GPU
            a_gpu = action if isinstance(action, torch.Tensor) else torch.as_tensor(action)
            a_gpu = a_gpu.to(device=self.device, dtype=torch.long)
            if a_gpu.ndim == 1:
                a_gpu = a_gpu.view(-1, 1)
            # Determine provisional batch size from obs/act
            n_gpu = max(
                o_prev_gpu.shape[0] if o_prev_gpu.ndim > 3 else 1,
                o_next_gpu.shape[0] if o_next_gpu.ndim > 3 else 1,
                a_gpu.shape[0] if a_gpu.ndim >= 2 else 1,
            )

            # Rewards (gracefully handle None by zero-filling)
            r_any = _get_with_fallback(next_td, ("next", "reward"), "reward")
            if r_any is None:
                r_gpu = torch.zeros(n_gpu, 1, device=self.device, dtype=torch.float32)
            else:
                r_gpu = (r_any if isinstance(r_any, torch.Tensor) else torch.as_tensor(r_any)).to(self.device, dtype=torch.float32)
                if r_gpu.ndim == 0:
                    r_gpu = r_gpu.view(1, 1)
                elif r_gpu.ndim == 1:
                    r_gpu = r_gpu.view(-1, 1)

            # Dones (gracefully handle None by zero-filling)
            d_any = _get_with_fallback(next_td, ("next", "done"), "done")
            if d_any is None:
                d_gpu = torch.zeros(n_gpu, 1, device=self.device, dtype=torch.float32)
            else:
                d_gpu = (d_any if isinstance(d_any, torch.Tensor) else torch.as_tensor(d_any)).to(self.device, dtype=torch.float32)
                if d_gpu.ndim == 0:
                    d_gpu = d_gpu.view(1, 1)
                elif d_gpu.ndim == 1:
                    d_gpu = d_gpu.view(-1, 1)

            # Final batch size alignment including r/d
            n_gpu = max(
                n_gpu,
                r_gpu.shape[0] if r_gpu.ndim >= 2 else 1,
                d_gpu.shape[0] if d_gpu.ndim >= 2 else 1,
            )

            if o_prev_gpu.ndim == 4 and o_prev_gpu.shape[0] == 1 and n_gpu > 1:
                o_prev_gpu = o_prev_gpu.repeat(n_gpu, 1, 1, 1)
            if o_next_gpu.ndim == 4 and o_next_gpu.shape[0] == 1 and n_gpu > 1:
                o_next_gpu = o_next_gpu.repeat(n_gpu, 1, 1, 1)
            if a_gpu.shape[0] != n_gpu:
                a_gpu = _col_gpu(a_gpu.squeeze(-1), n_gpu, torch.long)
            if r_gpu.shape[0] != n_gpu:
                r_gpu = _col_gpu(r_gpu.squeeze(-1), n_gpu, torch.float32)
            if d_gpu.shape[0] != n_gpu:
                d_gpu = _col_gpu(d_gpu.squeeze(-1), n_gpu, torch.float32)

            # Build TensorDict on GPU
            sample_gpu = TensorDict(
                {
                    "observation": o_prev_gpu,
                    "action": a_gpu,
                    "reward": r_gpu,
                    "done": d_gpu,
                    "next": {"observation": o_next_gpu},
                },
                batch_size=[n_gpu] if n_gpu > 1 else [],
                device=self.device,
            )

            # LLM annotations on GPU (optional)
            if llm_out is not None:
                pl = llm_out.get("prior_logits") if isinstance(llm_out, dict) else None
                cf = llm_out.get("confidence") if isinstance(llm_out, dict) else None
                ft = llm_out.get("features") if isinstance(llm_out, dict) else None
                if pl is not None:
                    sample_gpu.set("llm_prior_logits", torch.as_tensor(pl, device=self.device, dtype=torch.float32))
                if cf is not None:
                    sample_gpu.set("llm_confidence", torch.as_tensor(cf, device=self.device, dtype=torch.float32))
                if ft is not None and getattr(self.cfg.model, "llm_features_dim", 0) > 0:
                    sample_gpu.set("llm_features", torch.as_tensor(ft, device=self.device, dtype=torch.float32))

            try:
                if getattr(self, "_replay_lock", None) is not None:
                    with self._replay_lock:
                        self.replay.extend(sample_gpu)  # type: ignore[attr-defined]
                else:
                    self.replay.extend(sample_gpu)  # type: ignore[attr-defined]
            except Exception:
                # Fallback to per-row add if extend is unavailable
                if n_gpu > 1:
                    if getattr(self, "_replay_lock", None) is not None:
                        with self._replay_lock:
                            for i in range(n_gpu):
                                self.replay.add(sample_gpu[i])
                    else:
                        for i in range(n_gpu):
                            self.replay.add(sample_gpu[i])
                else:
                    if getattr(self, "_replay_lock", None) is not None:
                        with self._replay_lock:
                            self.replay.add(sample_gpu)
                    else:
                        self.replay.add(sample_gpu)
            return

        # Vectorized fast-path if batch > 1 (CPU replay path)
        if n > 1 and isinstance(prev_obs, torch.Tensor) and prev_obs.ndim > 3:
            # Helper: ensure (n,1) column tensor with desired dtype
            def _as_column(x: torch.Tensor, n_envs: int, dtype: torch.dtype | None = None) -> torch.Tensor:
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                if dtype is not None:
                    x = x.to(dtype)
                # Scalar -> repeat (n,1)
                if x.numel() == 1:
                    return x.view(1, 1).repeat(n_envs, 1)
                # 1D -> (n,1)
                if x.ndim == 1:
                    if x.shape[0] == n_envs:
                        return x.view(n_envs, 1)
                    if x.shape[0] == 1:
                        return x.view(1, 1).repeat(n_envs, 1)
                # 2D+ -> keep first column
                if x.ndim >= 2:
                    if x.shape[0] != n_envs:
                        x = x.view(n_envs, -1)
                    if x.shape[1] != 1:
                        x = x[:, :1]
                    return x
                # Fallback
                return x.view(n_envs, 1)
            # To uint8 batched
            def _to_uint8_b(x: torch.Tensor) -> torch.Tensor:
                if x.dtype == torch.uint8:
                    return x.cpu()
                return x.clamp(0, 1).mul(255).to(torch.uint8).cpu()

            o_prev_b = _to_uint8_b(prev_obs)
            o_next_b = _to_uint8_b(next_obs) if isinstance(next_obs, torch.Tensor) and next_obs.ndim > 3 else _to_uint8_b(prev_obs)

            a_b = _as_column(act if isinstance(act, torch.Tensor) else torch.as_tensor(act), n, dtype=torch.long)
            r_b = _as_column(rew if isinstance(rew, torch.Tensor) else torch.as_tensor(rew), n, dtype=torch.float32)
            d_b = _as_column(done if isinstance(done, torch.Tensor) else torch.as_tensor(done), n, dtype=torch.float32)

            sample_b = TensorDict(
                {
                    "observation": o_prev_b,
                    "action": a_b.cpu(),
                    "reward": r_b.cpu(),
                    "done": d_b.cpu(),
                    "next": {"observation": o_next_b},
                },
                batch_size=[n],
            )

            # Batched LLM annotations
            if llm_out is not None:
                pl = llm_out.get("prior_logits") if isinstance(llm_out, dict) else None
                cf = llm_out.get("confidence") if isinstance(llm_out, dict) else None
                ft = llm_out.get("features") if isinstance(llm_out, dict) else None
                if pl is not None:
                    sample_b.set("llm_prior_logits", torch.as_tensor(pl).cpu())
                if cf is not None:
                    sample_b.set("llm_confidence", torch.as_tensor(cf).cpu())
                if ft is not None and getattr(self.cfg.model, "llm_features_dim", 0) > 0:
                    sample_b.set("llm_features", torch.as_tensor(ft).cpu())

            # Single extend call
            try:
                if getattr(self, "_replay_lock", None) is not None:
                    with self._replay_lock:
                        self.replay.extend(sample_b)
                else:
                    self.replay.extend(sample_b)
            except Exception:
                # Fallback to add loop if backend lacks extend
                if getattr(self, "_replay_lock", None) is not None:
                    with self._replay_lock:
                        for i in range(n):
                            self.replay.add(sample_b[i])
                else:
                    for i in range(n):
                        self.replay.add(sample_b[i])
            return

        # Fallback: per-env add (single)
        o_prev = prev_obs if not (isinstance(prev_obs, torch.Tensor) and prev_obs.ndim > 3) else prev_obs[0]
        o_next = next_obs if not (isinstance(next_obs, torch.Tensor) and next_obs.ndim > 3) else next_obs[0]
        a_i = act if not (isinstance(act, torch.Tensor) and act.ndim > 1) else act[0]
        r_i = rew if not (isinstance(rew, torch.Tensor) and rew.ndim > 0) else rew[0]
        d_i = done if not (isinstance(done, torch.Tensor) and done.ndim > 0) else done[0]

        def _to_uint8(x: torch.Tensor) -> torch.Tensor:
            try:
                if x.dtype == torch.uint8:
                    return x.cpu()
                return x.clamp(0, 1).mul(255).to(torch.uint8).cpu()
            except Exception:
                return x.cpu()

        sample = TensorDict(
            {
                "observation": _to_uint8(o_prev) if isinstance(o_prev, torch.Tensor) else o_prev,
                "action": a_i.cpu() if isinstance(a_i, torch.Tensor) else torch.tensor(a_i),
                "reward": (r_i if isinstance(r_i, torch.Tensor) else torch.tensor(r_i)).to(torch.float32).cpu().view(1, -1),
                "done": (d_i if isinstance(d_i, torch.Tensor) else torch.tensor(d_i)).to(torch.float32).cpu().view(1, -1),
                "next": {"observation": _to_uint8(o_next) if isinstance(o_next, torch.Tensor) else o_next},
            },
            batch_size=[],
        )

        if llm_out is not None:
            pl = llm_out.get("prior_logits") if isinstance(llm_out, dict) else None
            cf = llm_out.get("confidence") if isinstance(llm_out, dict) else None
            ft = llm_out.get("features") if isinstance(llm_out, dict) else None
            if pl is not None:
                sample.set("llm_prior_logits", torch.as_tensor(pl).cpu())
            if cf is not None:
                sample.set("llm_confidence", torch.as_tensor(cf).cpu())
            if ft is not None and getattr(self.cfg.model, "llm_features_dim", 0) > 0:
                sample.set("llm_features", torch.as_tensor(ft).cpu())

        if getattr(self, "_replay_lock", None) is not None:
            with self._replay_lock:
                self.replay.add(sample)
        else:
            self.replay.add(sample)

    def _collect(self, steps: int):
        td = self._last_td
        # エピソード集計
        if not hasattr(self, "_ep_return"):
            self._ep_return = 0.0
            self._ep_len = 0
        if self._action_hist is None:
            try:
                import torch as _torch
                n = int(getattr(self.agent, "n_actions", 0))
                self._action_hist = _torch.zeros(n, dtype=_torch.long)
            except Exception:
                self._action_hist = None
        for _ in range(steps):
            obs = self._get_obs_tensor(td)
            # LLM 事前/特徴の取得（任意、失敗は無視） with throttling+cache
            llm_out = None
            h_for_act = None
            try:
                if getattr(self.cfg.train, "use_llm", False):
                    # Compute uncertainty & beta quickly from current policy (batch)
                    t_enc = time.perf_counter()
                    with torch.no_grad():
                        z = self.agent._enc_cache(obs)
                        h, _ = self.agent.world.rssm(
                            z.unsqueeze(1), torch.zeros(1, z.size(0), z.size(-1), device=z.device)
                        )
                        h = h.squeeze(1)  # [N,latent]
                        h_for_act = h
                        base_logits, uncertainty = self.agent.ac.policy_logits(
                            h, return_uncertainty=True
                        )
                        beta = self.agent.uncertainty_gate.compute_beta(uncertainty)  # [N]
                    log_event('collect.encode_rssm', 'gpu', (time.perf_counter() - t_enc) * 1000.0)

                    import numpy as _np
                    n = h.size(0)
                    num_actions = int(getattr(self.agent, "n_actions", 0))
                    # Prepare per-env outputs on GPU (default zeros)
                    prior_logits_pack = torch.zeros(n, num_actions, dtype=torch.float32, device=self.device)
                    features_dim = int(getattr(self.cfg.model, "llm_features_dim", 0))
                    features_pack = torch.zeros(n, features_dim, dtype=torch.float32, device=self.device) if features_dim > 0 else None
                    conf_pack = torch.zeros(n, 1, dtype=torch.float32, device=self.device)

                    # Pre-cache context config (build actual context lazily per selected env)
                    import os as _os
                    _dbg_on = _os.environ.get("LORE_DEBUG_LLM", "0") in ("1", "true", "True")
                    send_latent = bool(getattr(self.cfg.train, "llm_send_latent", True))
                    send_summary = bool(getattr(self.cfg.train, "llm_send_summary", True))
                    send_image = bool(getattr(self.cfg.train, "llm_send_image", True))
                    latent_dim = int(getattr(self.cfg.train, "llm_latent_dim", 32))
                    img_s = int(getattr(self.cfg.train, "llm_image_size", 16))
                    single_ch = bool(getattr(self.cfg.train, "llm_image_single_channel", True))
                    # latent projector (single shared) built on-demand
                    if send_latent and not hasattr(self, "_llm_latent_proj"):
                        in_d = h.size(-1)
                        self._llm_latent_proj = torch.nn.Linear(in_d, latent_dim).to(self.device).eval()

                    # Pre-build cache keys using GPU batch operations
                    cache_keys = [None] * n
                    cache_hashes = None
                    if self._llm_use_cache:
                        try:
                            # Batch normalize on GPU
                            v_batch = h.detach().float()  # [N, latent]
                            v_batch = v_batch / (v_batch.norm(dim=1, keepdim=True) + 1e-8)
                            q_batch = (v_batch * 32.0).clamp(-127, 127).to(torch.int8)
                            # Compute hash on GPU using simple sum (fast approximation)
                            cache_hashes = torch.sum(q_batch * torch.arange(1, q_batch.size(1)+1, device=q_batch.device), dim=1)
                            cache_hashes = cache_hashes.detach()  # Keep on GPU for now
                        except Exception:
                            cache_hashes = None

                    # Try cache for each env using GPU hash lookup
                    cached_mask = torch.zeros(n, dtype=torch.bool)
                    if cache_hashes is not None:
                        # Convert to hashable keys only when needed
                        for i in range(n):
                            try:
                                hash_key = int(cache_hashes[i].item())  # Single .item() call
                                if hash_key in self._llm_cache:
                                    out = self._llm_cache[hash_key]
                                    logits = torch.as_tensor(out.get("prior_logits"), dtype=torch.float32)
                                    if logits.numel() == num_actions:
                                        prior_logits_pack[i] = logits
                                        cached_mask[i] = True
                                        self._llm_cache_hits += 1
                                    else:
                                        self._llm_cache_misses += 1
                                else:
                                    self._llm_cache_misses += 1
                            except Exception:
                                self._llm_cache_misses += 1

                    # PriorNet fill for non-cached rows (usage rate計測)
                    if self._priornet is not None:
                        try:
                            with torch.no_grad():
                                pn_logits = self._priornet(h.detach())  # [N,A]
                            empty_rows = (prior_logits_pack.abs().sum(dim=1) == 0)
                            # opportunities: rows that were empty after cache
                            self._priornet_opportunities_total += int(empty_rows.sum().item())
                            # Keep PriorNet output on GPU
                            prior_logits_pack[empty_rows] = pn_logits[empty_rows].detach()
                            self._priornet_used_total += int(empty_rows.sum().item())
                        except Exception:
                            pass

                    # Select one env for actual API call per step (argmax beta among candidates)
                    if self._llm_call_budget > 0:
                        cooldown_ok = (self._llm_steps_since_call_env >= self._llm_cooldown)
                        # Interpret threshold as fraction of beta_max if <=1, else absolute
                        try:
                            bm = float(getattr(self.agent.uncertainty_gate, 'beta_max', 0.3))
                        except Exception:
                            bm = 0.3
                        thr = float(self._llm_beta_thr)
                        beta_threshold = thr * bm if thr <= 1.0 else thr
                        # Keep beta computation on GPU
                        beta_ok = (beta >= beta_threshold)
                        # Macro boundary mask: episode start or plateau (respect config)
                        call_on_ep = bool(getattr(self.cfg.train, "llm_call_on_episode_boundary", True))
                        boundary_mask = self._just_reset.clone() if call_on_ep else torch.zeros_like(self._just_reset)
                        plateau_N = int(getattr(self.cfg.train, "llm_plateau_frames", 10000))
                        boundary_mask |= (self._no_reward_steps >= plateau_N)
                        # Enforce minimum macro interval since last LLM call per env
                        try:
                            min_macro = int(getattr(self.cfg.train, "llm_min_macro_interval", 0) or 0)
                        except Exception:
                            min_macro = 0
                        if min_macro > 0:
                            boundary_mask &= (self._llm_steps_since_call_env >= min_macro)
                        # Prefer API when we have boundary and not cached (priornet may already fill, but API can overwrite one row for higher fidelity)
                        candidates = (~cached_mask) & cooldown_ok & beta_ok & boundary_mask
                        if candidates.any():
                            # Pick highest beta among candidates (GPU operations)
                            beta_sel = beta.masked_fill(~candidates, -1e9)
                            idx = int(torch.argmax(beta_sel).item())  # Only one .item() call
                            try:
                                # Build context for selected env only (GPU operations first)
                                ctx_i = None
                                try:
                                    item = {}
                                    if send_latent and hasattr(self, "_llm_latent_proj"):
                                        with torch.no_grad():
                                            lp_gpu = self._llm_latent_proj(h[idx:idx+1]).squeeze(0)  # Keep on GPU
                                        # Convert to list only when finalizing context
                                        item["latent"] = lp_gpu.detach().cpu().float()[:latent_dim].tolist()
                                    if send_summary:
                                        item["summary"] = {
                                            "step": int(self.global_step),
                                            "no_reward_steps": int(self._no_reward_steps[idx].item()) if idx < self._no_reward_steps.numel() else 0,
                                            "beta": float(beta[idx].item()) if idx < beta.numel() else 0.0,
                                        }
                                    if send_image:
                                        try:
                                            # downsample obs[idx]: [C,H,W] expected
                                            oi = obs[idx] if obs.dim() >= 4 else obs
                                            oi = torch.nn.functional.interpolate(oi.unsqueeze(0), size=(img_s, img_s), mode="area").squeeze(0)
                                            if single_ch and oi.dim() == 3 and oi.shape[0] > 1:
                                                oi = oi[:1]
                                            # Defer CPU conversion until context finalization
                                            oi_gpu = oi.clamp(0, 1).mul(255).to(torch.uint8)
                                            item["image"] = oi_gpu.cpu().numpy().tolist()  # small size, keep as nested list
                                        except Exception:
                                            pass
                                    ctx_i = {"items": [item]}
                                    if _dbg_on:
                                        print(f"[Trainer] LLM context built for idx={idx} latent_dim={latent_dim} img={img_s}x{img_s}")
                                except Exception:
                                    ctx_i = None
                                # Convert obs to numpy only at the last moment for LLM call
                                obs_i = obs[idx:idx+1].detach().cpu().numpy()
                                t_llm = time.perf_counter()
                                out = self.llm.infer(obs_i, num_actions=num_actions, context=ctx_i)
                                log_event('collect.llm_call', 'cpu', (time.perf_counter() - t_llm) * 1000.0)
                                mask = out.get("mask", _np.zeros((1,), dtype=_np.float32))
                                valid = bool((_np.asarray(mask).sum() > 0)) if mask is not None else True
                                import os as _os
                                if _os.environ.get("LORE_DEBUG_LLM", "0") in ("1","true","True"):
                                    try:
                                        lg = out.get("prior_logits")
                                        print(f"[Trainer] LLM call idx={idx} valid={valid} logits_len={len(lg) if lg is not None else 0}")
                                    except Exception:
                                        pass
                            except Exception as _e:
                                out = None
                                valid = False
                                try:
                                    import os as _os
                                    if _os.environ.get("LORE_DEBUG_LLM", "0") in ("1","true","True"):
                                        print(f"[Trainer] LLM call exception idx={idx} err='{str(_e)[:200]}'")
                                except Exception:
                                    pass
                            if valid and out is not None:
                                logits = torch.as_tensor(out.get("prior_logits"), dtype=torch.float32)
                                if logits.numel() == num_actions:
                                    prior_logits_pack[idx] = logits
                                    if features_pack is not None:
                                        feats = out.get("features")
                                        if feats is not None:
                                            fp = torch.as_tensor(feats, dtype=torch.float32)
                                            if fp.numel() == features_dim:
                                                features_pack[idx] = fp
                                    cf = out.get("confidence", _np.zeros((1,), dtype=_np.float32))
                                    conf_pack[idx] = torch.as_tensor(cf, dtype=torch.float32).view(1, 1)
                                    # Update counters
                                    self._llm_intervals.append(int(self._llm_steps_since_call_env[idx].item()))
                                    self._llm_beta_when_call.append(float(beta[idx].item()))
                                    self._llm_calls_total += 1
                                    self._llm_call_budget -= 1
                                    self._llm_steps_since_call_env[idx] = 0
                                    self._llm_calls_env[idx] += 1
                                    # Cache store using GPU hash
                                    if cache_hashes is not None:
                                        hash_key = int(cache_hashes[idx].item())
                                        self._llm_cache[hash_key] = out
                                else:
                                    # Failure: cooldown applies even if invalid to avoid spamming
                                    self._llm_errors_total += 1
                                    self._llm_steps_since_call_env[idx] = 0
                            else:
                                # Failure: cooldown applies even if exception
                                self._llm_errors_total += 1
                                self._llm_steps_since_call_env[idx] = 0

                    # If cache miss and priornet available, try priornet before API
                    if self._priornet is not None:
                        try:
                            with torch.no_grad():
                                pn_logits = self._priornet(h.detach())  # [N,A]
                            # Only fill where still zeros (no cache/API)
                            empty_rows = (prior_logits_pack.abs().sum(dim=1) == 0)
                            prior_logits_pack[empty_rows] = pn_logits[empty_rows].detach().cpu()
                        except Exception:
                            pass

                    # Package llm_out for replay: as dict of batched tensors
                    llm_out = {
                        "prior_logits": prior_logits_pack,
                        "features": features_pack if features_pack is not None else None,
                        "confidence": conf_pack,
                    }
            except Exception:
                llm_out = None

            # 初期ランダム行動期間
            with torch.no_grad():
                if self.frames_seen < self._random_warmup_steps:
                    import torch as _torch
                    n = int(getattr(self.agent, "n_actions", 1))
                    batch_n = int(obs.shape[0]) if obs.dim() >= 4 else 1
                    t_rand = time.perf_counter()
                    action = _torch.randint(low=0, high=n, size=(batch_n, 1), device=self.device)
                    log_event('collect.rand_action', 'gpu', (time.perf_counter() - t_rand) * 1000.0)
                else:
                    # Reuse precomputed h from LLM processing if available
                    t_act = time.perf_counter()
                    if h_for_act is not None:
                        action = self.agent.act(obs, precomputed_h=h_for_act)
                    else:
                        action = self.agent.act(obs)
                    log_event('collect.act', 'gpu', (time.perf_counter() - t_act) * 1000.0)
            # TorchRL envs expect action within a tensordict
            td.set("action", action.cpu())
            t_env = time.perf_counter()
            td = self.env.step(td)
            log_event('collect.env_step', 'cpu', (time.perf_counter() - t_env) * 1000.0)

            # 集計（報酬・探索ボーナス適用前の平均も使う）
            try:
                # TorchRL は step 後の値を ("next", key) に格納する実装が多い
                if ("next", "reward") in td.keys(True, True):
                    r = float(td.get(("next", "reward")).mean().item())
                elif "reward" in td.keys(True, True):
                    r = float(td.get("reward")).mean().item()
                else:
                    r = 0.0
            except Exception:
                r = 0.0
            # Intrinsic reward
            if self._rnd is not None:
                try:
                    obs_i = self._get_obs_tensor(td)
                    ri = self._rnd.intrinsic_reward(obs_i).mean().item()
                    if getattr(self.cfg.train, "intrinsic_norm", True):
                        # simple running normalization via EMA
                        if not hasattr(self, "_ri_mean"):
                            self._ri_mean = 0.0
                            self._ri_var = 1.0
                        m = 0.99
                        self._ri_mean = m * self._ri_mean + (1 - m) * ri
                        self._ri_var = m * self._ri_var + (1 - m) * (ri - self._ri_mean) ** 2
                        ri_n = (ri - self._ri_mean) / (self._ri_var ** 0.5 + 1e-6)
                    else:
                        ri_n = ri
                    r += float(self.cfg.train.intrinsic_coef) * ri_n
                    # Train predictor with configurable frequency (default every 4 steps)
                    intrinsic_update_every = int(getattr(self.cfg.train, "intrinsic_update_every", 4))
                    if self.frames_seen % intrinsic_update_every == 0:
                        _ = self._rnd.update(obs_i)
                except Exception:
                    pass
            # Episode-wise accumulators (per-env)
            try:
                rew_vec = td.get(("next", "reward")) if ("next", "reward") in td.keys(True, True) else td.get("reward")
                if rew_vec is not None and hasattr(self, "_ep_ret_env") and self._ep_ret_env is not None:
                    import torch as _torch
                    rv = rew_vec.view(-1).to(dtype=_torch.float32)
                    self._ep_ret_env[: rv.numel()] += rv
                    self._ep_len_env[: rv.numel()] += 1
                    # success key if provided, else positive reward heuristic
                    suc = td.get("success") if "success" in td.keys(True, True) else None
                    if suc is not None:
                        self._ep_success_env[: rv.numel()] |= suc.view(-1).to(dtype=_torch.bool)
                    else:
                        self._ep_success_env[: rv.numel()] |= (rv > 0)
            except Exception:
                pass
            # 行動ヒストグラム
            try:
                if self._action_hist is not None:
                    a = int(action.squeeze().item())
                    if 0 <= a < self._action_hist.numel():
                        self._action_hist[a] += 1
            except Exception:
                pass

            # エピソード終端時は自動リセット（ベクトル環境対応: ここでは一括リセットの簡易実装）
            try:
                dn = None
                if ("next", "done") in td.keys(True, True):
                    dn = td.get(("next", "done"))
                elif "done" in td.keys(True, True):
                    dn = td.get("done")
                if dn is not None:
                    done_mask = dn.to(dtype=torch.bool).view(-1)
                else:
                    done_mask = None
            except Exception:
                done_mask = None
            if done_mask is not None and done_mask.any():
                # Mark boundaries per-env
                self._just_reset[done_mask] = True
                self._no_reward_steps[done_mask] = 0
                # Log episode returns, success rate, and Crafter metrics if info提供
                try:
                    import torch as _torch
                    idxs = _torch.nonzero(done_mask, as_tuple=False).view(-1).cpu().tolist()
                    for _i in idxs:
                        if self._ep_ret_env is not None:
                            ret_i = float(self._ep_ret_env[_i].item())
                            self.logger.add_scalar("env/episode_return", ret_i, self.frames_seen)
                        # success_rate は Crafter 公式仕様に準拠して算出（達成タスク割合）
                        # ここではエピソード終端時の info から達成タスクを抽出し、そのエピソードの成功率を記録する
                        try:
                            def _extract_achievements_for_env(td_any, env_index: int):
                                info_obj = None
                                if ("next", "info") in td_any.keys(True, True):
                                    info_obj = td_any.get(("next", "info"))
                                elif "info" in td_any.keys(True, True):
                                    info_obj = td_any.get("info")
                                # envごとに取り出し
                                if isinstance(info_obj, (list, tuple)):
                                    if 0 <= env_index < len(info_obj) and isinstance(info_obj[env_index], dict):
                                        return info_obj[env_index].get("achievements") or info_obj[env_index].get("achievements/boolean")
                                # TensorDict 形式などのベストエフォート
                                try:
                                    from tensordict import TensorDict as _TD  # type: ignore
                                    if isinstance(info_obj, _TD):
                                        d = {k: v for k, v in info_obj.items() if isinstance(v, (int, float, bool))}
                                        if d:
                                            return d
                                except Exception:
                                    pass
                                if isinstance(info_obj, dict):
                                    return info_obj.get("achievements") or info_obj.get("achievements/boolean") or info_obj
                                return None

                            ach = _extract_achievements_for_env(td, _i)
                            if isinstance(ach, dict) and len(ach) > 0:
                                total_keys = 22 if 22 >= len(ach) else len(ach)
                                achieved = sum(1 for v in ach.values() if bool(v))
                                success_rate = achieved / float(total_keys)
                                self.logger.add_scalar("env/success_rate", float(success_rate), self.frames_seen)
                                self.logger.add_scalar("env/score_percent", float(success_rate * 100.0), self.frames_seen)
                        except Exception:
                            pass
                        # Crafter achievements aggregate（幾何平均スコア）は補助として記録（success_rateは上書きしない）
                        try:
                            info_any = None
                            if ("next", "info") in td.keys(True, True):
                                info_any = td.get(("next", "info"))
                            elif "info" in td.keys(True, True):
                                info_any = td.get("info")
                            ach_dict: dict[str, bool] | None = None
                            if isinstance(info_any, dict):
                                ach_dict = info_any.get("achievements") or info_any.get("achievements/boolean") or None
                            if ach_dict:
                                for k, v in ach_dict.items():
                                    if v:
                                        self._ach_counts[k] = self._ach_counts.get(k, 0) + 1
                                self._ach_episodes += 1
                                total_keys = len(self._ach_counts)
                                if total_keys > 0 and self._ach_episodes > 0:
                                    import math as _math
                                    gm = 1.0
                                    for c in self._ach_counts.values():
                                        p = max(1e-6, c / float(self._ach_episodes))
                                        gm *= p
                                    crafter_score = gm ** (1.0 / total_keys)
                                    self.logger.add_scalar("env/crafter_score", float(crafter_score), self.frames_seen)
                        except Exception:
                            pass
                        # reset per-env accumulators
                        if self._ep_ret_env is not None:
                            self._ep_ret_env[_i] = 0.0
                        if self._ep_len_env is not None:
                            self._ep_len_env[_i] = 0
                        if self._ep_success_env is not None:
                            self._ep_success_env[_i] = False
                except Exception:
                    pass
                # Try per-env reset
                try:
                    idx = torch.nonzero(done_mask, as_tuple=False).view(-1).cpu().tolist()
                    # Attempt various API signatures
                    td_reset = None
                    try:
                        td_reset = self.env.reset(env_ids=idx)  # type: ignore
                    except Exception:
                        try:
                            td_reset = self.env.reset(reset_envs=idx)  # type: ignore
                        except Exception:
                            try:
                                mask = done_mask.view(1, -1) if done_mask.dim() == 1 else done_mask
                                td_reset = self.env.reset(mask=mask)  # type: ignore
                            except Exception:
                                td_reset = self.env.reset()
                    if td_reset is not None:
                        td = td_reset
                except Exception:
                    pass

            t_rep = time.perf_counter()
            self._append_to_replay(self._last_td, action, td, llm_out)
            log_event('collect.replay_add', 'cpu', (time.perf_counter() - t_rep) * 1000.0)
            self._last_td = td

            self.global_step += 1
            # 合算フレーム数を加算（ベクトル環境のサンプル数）
            try:
                rew_vec = td.get(("next", "reward")) if ("next", "reward") in td.keys(True, True) else td.get("reward")
                if rew_vec is not None:
                    env_batch = int(rew_vec.view(-1).numel())
                else:
                    env_batch = int(self._n_envs)
            except Exception:
                env_batch = int(self._n_envs)
            self.frames_seen += env_batch
            hw_set_step(self.frames_seen)
            # Update per-env trackers
            try:
                rew_vec = td.get(("next", "reward")) if ("next", "reward") in td.keys(True, True) else td.get("reward")
                if rew_vec is not None:
                    rew_vec = rew_vec.view(-1)
                    self._no_reward_steps = torch.where(rew_vec != 0.0, torch.zeros_like(self._no_reward_steps), self._no_reward_steps + 1)
            except Exception:
                pass
            self._llm_steps_since_call_env += 1

            # 予算到達で早期停止（合算フレーム基準）
            try:
                if self.frames_seen >= int(getattr(self.cfg.train, "total_frames", 10**9)):
                    break
            except Exception:
                pass

            # Distill PriorNet occasionally using the most recent h/logits pairs
            try:
                self._priornet_last_update += 1
                upd_every = int(getattr(self.cfg.train, "llm_priornet_update_every", 50))
                if self._priornet is not None and self._priornet_opt is not None and (self._priornet_last_update % upd_every == 0):
                    # Use current batch h as inputs; targets are prior_logits_pack (where non-zero)
                    temp = float(getattr(self.cfg.train, "llm_priornet_temp", 2.0))
                    with torch.no_grad():
                        z = self.agent._enc_cache(obs)
                        h_cur, _ = self.agent.world.rssm(
                            z.unsqueeze(1), torch.zeros(1, z.size(0), z.size(-1), device=z.device)
                        )
                        h_cur = h_cur.squeeze(1)
                    logits_pred = self._priornet(h_cur)
                    targets = llm_out["prior_logits"].to(self.device)
                    mask_nonzero = (targets.abs().sum(dim=1) > 0).float().unsqueeze(1)
                    if mask_nonzero.sum() > 0:
                        import torch.nn.functional as F
                        p = F.log_softmax(logits_pred / temp, dim=-1)
                        q = F.softmax(targets / temp, dim=-1)
                        loss = -(q * p).sum(dim=-1)
                        loss = (loss * mask_nonzero.squeeze(1)).sum() / (mask_nonzero.sum() + 1e-6)
                        self._priornet_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        self._priornet_opt.step()
                        try:
                            self.logger.add_scalar("llm/priornet_distill_loss", float(loss.detach().cpu()), self.global_step)
                        except Exception:
                            pass
            except Exception:
                pass

    def _update(self, updates: int):
        logs = {}
        # Async path: prefetch CPU->GPU while agent is updating
        use_async = bool(getattr(self.cfg.train, "async_update", True))
        if use_async:
            try:
                from collections import deque
                import threading
                q: deque = deque(maxlen=2)
                stop_flag = {"v": False}
                logs_local = {}

                def loader_job():
                    while not stop_flag["v"]:
                        try:
                            t_samp = time.perf_counter()
                            if getattr(self, "_replay_lock", None) is not None:
                                with self._replay_lock:
                                    batch = self.replay.sample(self.cfg.train.batch_size)
                            else:
                                batch = self.replay.sample(self.cfg.train.batch_size)
                            log_event('update.sample', 'cpu', (time.perf_counter() - t_samp) * 1000.0)
                        except Exception:
                            break
                        # pin + async HtoD
                        try:
                            if hasattr(batch, "pin_memory"):
                                batch = batch.pin_memory()
                            t_to = time.perf_counter()
                            batch = batch.to(self.device, non_blocking=True)
                            log_event('update.host_to_device', 'dma', (time.perf_counter() - t_to) * 1000.0)
                        except Exception:
                            pass
                        q.append(batch)
                        if len(q) >= q.maxlen:
                            # backpressure: small sleep
                            time.sleep(0.001)

                loader = threading.Thread(target=loader_job, daemon=True)
                loader.start()

                iters = 0
                while iters < updates:
                    if not q:
                        time.sleep(0.001)
                        continue
                    batch = q.popleft()
                    # Decode uint8 observations to float on GPU
                    try:
                        if "observation" in batch.keys(True, True):
                            obs_t = batch.get("observation")
                            if obs_t.dtype == torch.uint8:
                                batch.set("observation", obs_t.to(torch.float32).div_(255.0))
                        if ("next", "observation") in batch.keys(True, True):
                            nxt = batch.get(("next", "observation"))
                            if nxt.dtype == torch.uint8:
                                batch.set(("next", "observation"), nxt.to(torch.float32).div_(255.0))
                    except Exception:
                        pass
                    # PER weights pass-through
                    try:
                        iw = None
                        if '_weights' in batch.keys(True, True):
                            iw = batch.get('_weights').to(self.device)
                            if iw.dim() == 1:
                                iw = iw.view(-1, 1)
                            batch.set('_sample_weights', iw)
                    except Exception:
                        pass
                    t_upd = time.perf_counter()
                    out = self.agent.update(batch)
                    log_event('update.agent', 'gpu', (time.perf_counter() - t_upd) * 1000.0)
                    logs_local.update(out)
                    iters += 1

                stop_flag["v"] = True
                loader.join(timeout=0.1)
                return logs_local
            except Exception:
                pass

        # Fallback: synchronous
        for _ in range(updates):
            try:
                t_samp = time.perf_counter()
                if getattr(self, "_replay_lock", None) is not None:
                    with self._replay_lock:
                        batch = self.replay.sample(self.cfg.train.batch_size)
                else:
                    batch = self.replay.sample(self.cfg.train.batch_size)
                log_event('update.sample', 'cpu', (time.perf_counter() - t_samp) * 1000.0)
            except Exception:
                # 収集が追いつかない場合は短い待機を挟み、即座に次の収集へ戻す
                time.sleep(0.001)
                break
            # move to device with non-blocking transfer (pin CPU memory first if possible)
            try:
                if hasattr(batch, "pin_memory"):
                    batch = batch.pin_memory()
                # Try bulk transfer if supported
                t_to = time.perf_counter()
                batch = batch.to(self.device, non_blocking=True)
                log_event('update.host_to_device', 'dma', (time.perf_counter() - t_to) * 1000.0)
            except Exception:
                # Fallback to individual key transfer
                for k in ["observation", "action", "reward"]:
                    if k in batch.keys(True, True):
                        t = batch.get(k)
                        try:
                            t = t.pin_memory() if hasattr(t, 'pin_memory') else t
                        except Exception:
                            pass
                        t0 = time.perf_counter()
                        batch.set(k, t.to(self.device, non_blocking=True))
                        log_event('update.host_to_device_key', 'dma', (time.perf_counter() - t0) * 1000.0, extras={'key': k})

            # Decode uint8 observations to float on GPU
            try:
                if "observation" in batch.keys(True, True):
                    obs_t = batch.get("observation")
                    if obs_t.dtype == torch.uint8:
                        batch.set("observation", obs_t.to(torch.float32).div_(255.0))
                if ("next", "observation") in batch.keys(True, True):
                    nxt = batch.get(("next", "observation"))
                    if nxt.dtype == torch.uint8:
                        batch.set(("next", "observation"), nxt.to(torch.float32).div_(255.0))
            except Exception:
                pass
            # PER: importance weights support
            try:
                iw = None
                if '_weights' in batch.keys(True, True):
                    iw = batch.get('_weights').to(self.device)
                    # 形状合わせ（[B]→[B,1]）
                    if iw.dim() == 1:
                        iw = iw.view(-1, 1)
                    # 重要度はメトリクスとしても記録
                    try:
                        self.logger.add_scalar('replay/importance_mean', float(iw.mean().detach().cpu()), self.frames_seen)
                    except Exception:
                        pass
                # 一時的にエージェントへ重みを渡す（バッチに格納）
                if iw is not None:
                    batch.set('_sample_weights', iw)
            except Exception:
                pass

            t_upd = time.perf_counter()
            out = self.agent.update(batch)
            log_event('update.agent', 'gpu', (time.perf_counter() - t_upd) * 1000.0)
            logs.update(out)
            # PER: 可能ならTD誤差で優先度更新
            try:
                if hasattr(self.replay, 'update_priorities') and 'value/td_abs_mean' in out:
                    # 代表値でなく、バッチ個別誤差が望ましいが簡易版として平均誤差を使用
                    err = out['value/td_abs_mean']
                    bs = int(self.cfg.train.batch_size)
                    errs = torch.full((bs,), float(err), dtype=torch.float32)
                    idx = batch.get('_indices') if '_indices' in batch.keys(True, True) else None
                    if idx is not None:
                        self.replay.update_priorities(idx, errs)
            except Exception:
                pass
        return logs

    def train(self):
        total_frames = int(self.cfg.train.total_frames)
        # Use smaller chunks to interleave collect<->update more tightly
        default_collect = int(self.cfg.train.collect_steps_per_iter)
        default_update = int(self.cfg.train.updates_per_collect)
        collect_per_iter = int(getattr(self.cfg.train, 'collect_chunk_steps', default_collect))
        updates_per_collect = int(getattr(self.cfg.train, 'update_chunk_steps', default_update))

        # 合算フレーム数で学習終了を判定
        # Optional async collector thread keeps CPU env stepping while GPU updates
        use_async_collect = bool(getattr(self.cfg.train, 'async_collect', True))
        collector_thread = None
        stop_collect = {'v': False}

        def _collector_loop():
            try:
                while not stop_collect['v'] and self.frames_seen < total_frames:
                    self._collect(collect_per_iter)
            except Exception:
                pass

        if use_async_collect:
            try:
                import threading as _threading
                collector_thread = _threading.Thread(target=_collector_loop, daemon=True)
                collector_thread.start()
            except Exception:
                collector_thread = None

        while self.frames_seen < total_frames:
            # Anneal exploration parameters
            try:
                # Entropy coefficient: linear anneal from start -> entropy_anneal_to
                ent_start = float(getattr(self.cfg.train, "entropy_coef", 0.05))
                ent_to = float(getattr(self.cfg.train, "entropy_anneal_to", ent_start))
                ent_frames = max(1, int(getattr(self.cfg.train, "entropy_anneal_frames", total_frames)))
                frac = min(1.0, self.frames_seen / ent_frames)
                current_entropy_coef = ent_start + (ent_to - ent_start) * frac
                if hasattr(self.agent, "entropy_coef"):
                    self.agent.entropy_coef = float(current_entropy_coef)
                # Epsilon-greedy: linear decay from start -> epsilon_greedy_decay_to
                eps_start = float(getattr(self.cfg.train, "epsilon_greedy", 0.2))
                eps_to = float(getattr(self.cfg.train, "epsilon_greedy_decay_to", eps_start))
                eps_frames = max(1, int(getattr(self.cfg.train, "epsilon_greedy_decay_frames", total_frames)))
                frac_e = min(1.0, self.frames_seen / eps_frames)
                current_eps = eps_start + (eps_to - eps_start) * frac_e
                if hasattr(self.agent, "epsilon_greedy"):
                    self.agent.epsilon_greedy = float(current_eps)
                # Log occasionally
                if self.global_step % self._log_interval == 0:
                    try:
                        self.logger.add_scalar("exploration/entropy_coef", float(current_entropy_coef), self.frames_seen)
                        self.logger.add_scalar("exploration/epsilon_greedy", float(current_eps), self.frames_seen)
                    except Exception:
                        pass
            except Exception:
                pass
            # Budget-aware annealing (saver mode) + long-run schedule to hit target calls/M
            try:
                total_budget = max(1, int(getattr(self.cfg.train, "llm_call_budget_total", 1)))
                left = max(0, int(self._llm_call_budget))
                r = left / total_budget
                base_thr = float(getattr(self.cfg.train, "llm_beta_call_threshold", 0.65))
                base_cd = int(getattr(self.cfg.train, "llm_cooldown_steps", 2000))
                if r < 0.2:
                    self._llm_beta_thr = base_thr + 0.10
                    self._llm_cooldown = max(self._llm_cooldown, int(base_cd * 1.25 * 1.25))
                elif r < 0.4:
                    self._llm_beta_thr = base_thr + 0.05
                    self._llm_cooldown = max(self._llm_cooldown, int(base_cd * 1.25))

                # For very long runs (e.g., 10M), anneal target calls per million
                target_cpm = max(1, int(getattr(self.cfg.train, "llm_calls_per_million", 500)))
                # steps per planned call based on achieved calls so far
                steps = max(1, int(self.frames_seen))
                calls = max(1, int(self._llm_calls_total))
                achieved_spc = steps / calls if calls > 0 else float("inf")
                target_spc = 1_000_000 / float(target_cpm)
                # If we are calling too often (< target_spc), harden thresholds slightly
                if achieved_spc < target_spc * 0.9:
                    self._llm_beta_thr = min(1.0, self._llm_beta_thr + 0.02)
                    self._llm_cooldown = max(self._llm_cooldown, int(base_cd * 1.1))
                # If we are too conservative (> 2x target), soften slightly early on
                elif achieved_spc > target_spc * 2.0 and steps < int(0.3 * total_frames):
                    self._llm_beta_thr = max(0.0, self._llm_beta_thr - 0.02)
                    self._llm_cooldown = max(1, int(self._llm_cooldown * 0.9))
            except Exception:
                pass

            if not use_async_collect:
                self._collect(collect_per_iter)
            logs = self._update(updates_per_collect)
            if logs:
                self._last_logs = logs
            # logging (whitelisted) with catch-up to avoid missing due to big frame jumps
            if self.frames_seen >= self._next_log_step:
                to_log = self._last_logs if self._last_logs else logs
                if to_log:
                    for k, v in to_log.items():
                        self.logger.add_scalar(k, float(v), self.frames_seen)
                # 補助: 平均エピソードリターン（短窓）
                try:
                    import torch as _torch
                    if hasattr(self, "_ep_ret_env") and self._ep_ret_env is not None and hasattr(self, "_ep_len_env"):
                        lens = self._ep_len_env.clamp_min(1).to(dtype=_torch.float32)
                        mean_ret = float((self._ep_ret_env / lens).mean().item())
                        self.logger.add_scalar("env/mean_episode_return", mean_ret, self.frames_seen)
                except Exception:
                    pass
                # advance next log step; catch up if we jumped multiple intervals
                while self._next_log_step <= self.frames_seen:
                    self._next_log_step += self._log_interval
            # LLM系ログはダッシュボード対象外なので抑制
            # env/reward の代わりにエピソードリターンを用いる（別途終端時に出力）
            # checkpoint（任意、エージェント側の save 実装に委ねる）
            try:
                if self._save_every and (self.frames_seen % self._save_every == 0):
                    if hasattr(self.agent, "save"):
                        import os
                        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
                        path = f"{self.cfg.ckpt_dir}/ckpt_step_{self.frames_seen}.pt"
                        self.agent.save(path)
            except Exception:
                pass
            # GPU monitoring and logging (detailed) with catch-up
            if self._gpu_monitoring and torch.cuda.is_available():
                if self.frames_seen >= self._next_gpu_report:
                    try:
                        dev = torch.cuda.current_device()
                        name = torch.cuda.get_device_name(dev)
                        props = torch.cuda.get_device_properties(dev)
                        allocated = torch.cuda.memory_allocated(dev) / 1024**2
                        reserved = torch.cuda.memory_reserved(dev) / 1024**2
                        max_alloc = torch.cuda.max_memory_allocated(dev) / 1024**2
                        # Utilization via nvidia-smi
                        import subprocess
                        util = 'N/A'
                        mem_util = 'N/A'
                        try:
                            q = ['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory','--format=csv,noheader,nounits']
                            r = subprocess.run(q, capture_output=True, text=True, timeout=1)
                            if r.returncode == 0 and r.stdout:
                                parts = r.stdout.strip().split(',')
                                if len(parts) >= 2:
                                    util = parts[0].strip()+'%'
                                    mem_util = parts[1].strip()+'%'
                        except Exception:
                            pass
                        # Log scalars
                        self.logger.add_scalar("gpu/alloc_mb", allocated, self.frames_seen)
                        self.logger.add_scalar("gpu/reserved_mb", reserved, self.frames_seen)
                        self.logger.add_scalar("gpu/max_alloc_mb", max_alloc, self.frames_seen)
                        # Print brief status
                        print(f"[GPU] step={self.frames_seen} dev={dev} {name} SMs={props.multi_processor_count} mem={allocated:.0f}/{reserved:.0f}MB util={util} mem_util={mem_util}")
                    except Exception:
                        pass
                    # advance next gpu report; catch up if we jumped multiple intervals
                    while self._next_gpu_report <= self.frames_seen:
                        self._next_gpu_report += 500
            
            # Update logger step and flush with rate limiting
            self.logger.set_current_step(self.frames_seen)
            self.logger.flush()

            # Heartbeat every 1000 steps exactly
            try:
                if hasattr(self.logger, 'maybe_step'):
                    self.logger.maybe_step(self.frames_seen)
            except Exception:
                pass

        # stop collector
        if collector_thread is not None:
            stop_collect['v'] = True
            try:
                collector_thread.join(timeout=0.2)
            except Exception:
                pass

