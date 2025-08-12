from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ..utils.replay import make_replay
try:
    from ..utils.synthetic_replay import EnhancedReplayBuffer  # type: ignore
except Exception:  # pragma: no cover
    EnhancedReplayBuffer = None  # type: ignore
from ..utils.rnd import RND  # type: ignore
try:
    from ..utils.llm_adapter import LLMAdapter, LLMAdapterConfig  # type: ignore
except Exception:  # pragma: no cover
    LLMAdapter = None  # type: ignore
    LLMAdapterConfig = None  # type: ignore
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore
try:
    from ..envs.crafter_env import make_crafter_env  # type: ignore
except Exception:  # pragma: no cover
    make_crafter_env = None  # type: ignore


@dataclass
class _Stats:
    frames_total: int = 0
    updates_total: int = 0
    last_update_loss: Optional[float] = None
    replay_size: int = 0
    last_update_ms: Optional[float] = None
    episodes_total: int = 0
    score_percents: list = field(default_factory=list)
    episode_returns: list = field(default_factory=list)
    episode_returns_ext: list = field(default_factory=list)
    last_entropy: Optional[float] = None
    last_value_ev: Optional[float] = None
    last_td_abs: Optional[float] = None
    last_psnr: Optional[float] = None
    last_grad_norm: Optional[float] = None
    last_step_reward_mean: Optional[float] = None
    last_step_reward_ext_mean: Optional[float] = None
    last_step_intrinsic_mean: Optional[float] = None


class AsyncTrainer:
    """Parallel trainer: CPU env collector + GPU learner run concurrently.

    - Collector thread: steps environments on CPU and appends to replay
    - Loader thread: samples mini-batches and moves them to GPU in advance
    - Learner (main thread): consumes preloaded batches and runs updates

    Status is printed periodically (no external logger required).
    """

    def __init__(self, env, agent, logger, cfg, device) -> None:  # noqa: D401
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.device = device
        self.logger = logger  # kept for interface compatibility (No-Op by design)

        # Replay buffer (CPU) — prefer EnhancedReplayBuffer if available when synthetic is enabled
        use_synth = bool(getattr(cfg.train, 'synthetic_execution_prob', 0.0)) or bool(getattr(cfg.train, 'synthetic_ratio_max', 0.0))
        if EnhancedReplayBuffer is not None and use_synth:
            try:
                self.replay = EnhancedReplayBuffer(
                    capacity=int(getattr(cfg.train, 'replay_capacity', 100_000)),
                    synthetic_ratio_max=float(getattr(cfg.train, 'synthetic_ratio_max', 0.25)),
                    bc_regularization_coeff=float(getattr(cfg.model, 'lambda_bc', 0.1)),
                    importance_sampling=True,
                    synthetic_weight_decay=float(getattr(cfg.train, 'synthetic_weight_decay', 0.99)),
                )
            except Exception:
                self.replay = make_replay(
                    capacity=int(getattr(cfg.train, "replay_capacity", 100_000)),
                    use_priority=bool(getattr(cfg.train, "use_priority_replay", False)),
                )
        else:
            self.replay = make_replay(
                capacity=int(getattr(cfg.train, "replay_capacity", 100_000)),
                use_priority=bool(getattr(cfg.train, "use_priority_replay", False)),
            )

        # Specs
        try:
            self.obs_key = "pixels" if "pixels" in env.observation_spec.keys(True, True) else "observation"
        except Exception:
            self.obs_key = "observation"

        # Initial reset
        try:
            self._last_td = self.env.reset()
        except Exception:
            self._last_td = None

        # Episode accumulators (per-env)
        self._n_envs = 1
        try:
            if self._last_td is not None:
                bs = self._last_td.batch_size
                self._n_envs = int(bs[0]) if len(bs) > 0 else 1
        except Exception:
            self._n_envs = 1
        try:
            self._ep_ret_env = torch.zeros(self._n_envs, dtype=torch.float32)
            self._ep_len_env = torch.zeros(self._n_envs, dtype=torch.long)
            self._ep_ret_ext_env = torch.zeros(self._n_envs, dtype=torch.float32)
        except Exception:
            self._ep_ret_env = None
            self._ep_len_env = None
            self._ep_ret_ext_env = None

        # LLM adapter (optional)
        self._llm_enabled = bool(getattr(self.cfg.train, "use_llm", False))
        self.llm = None
        try:
            if self._llm_enabled and LLMAdapter is not None and LLMAdapterConfig is not None:
                self.llm = LLMAdapter(
                    LLMAdapterConfig(
                        enabled=True,
                        model=getattr(self.cfg.train, "llm_model", "gemini-2.5-flash-lite"),
                        features_dim=getattr(self.cfg.model, "llm_features_dim", 0),
                        use_cli=getattr(self.cfg.train, "llm_use_cli", False),
                        timeout_s=getattr(self.cfg.train, "llm_timeout_s", 2.5),
                        api_retries=int(getattr(self.cfg.train, "llm_api_retries", 2)),
                    )
                )
        except Exception:
            self.llm = None
            self._llm_enabled = False
        # LLM throttling state
        import torch as _torch  # local alias
        self._llm_steps_since_call_env = _torch.full((self._n_envs,), 10**9, dtype=_torch.long)
        self._llm_call_budget = int(getattr(self.cfg.train, "llm_call_budget_total", 0) or 0)
        self._llm_cooldown = int(getattr(self.cfg.train, "llm_cooldown_steps", 0) or 0)
        self._llm_beta_thr = float(getattr(self.cfg.train, "llm_beta_call_threshold", 1.0))

        # Concurrency
        self._stop = {"v": False}
        self._lock = threading.Lock()
        self._stats = _Stats()

        # Bounded GPU batch queue (double-buffer)
        from collections import deque

        self._batch_queue: deque = deque(maxlen=4)
        self._batch_lock = threading.Lock()

        # Training schedule
        self.total_frames = int(getattr(cfg.train, "total_frames", 1_000_000))
        self.batch_size = int(getattr(cfg.train, "batch_size", 512))
        # Favor collect-heavy early then learner-heavy later
        self.collect_chunk = int(getattr(cfg.train, "collect_chunk_steps", 512))
        self.update_chunk = int(getattr(cfg.train, "update_chunk_steps", 128))
        self.min_replay_init = int(getattr(cfg.train, 'min_replay_init', getattr(cfg.train, 'init_random_frames', 10000)))
        self.init_random_frames = int(getattr(cfg.train, 'init_random_frames', 10000))
        # 初期softmax探索は総ステップの10-20%推奨（デフォルト: 200k）
        self.exploration_softmax_frames = int(getattr(cfg.train, 'exploration_softmax_frames', 200000))
        self.softmax_temp_start = float(getattr(cfg.train, 'softmax_temp_start', 1.35))
        self.softmax_temp_end = float(getattr(cfg.train, 'softmax_temp_end', 1.08))
        # 後半の詰み回避用に低温softmaxを稀に再導入
        self.late_softmax_temp = float(getattr(cfg.train, 'late_softmax_temp', 1.03))
        self.late_softmax_prob = float(getattr(cfg.train, 'late_softmax_prob', 0.15))

        # Exploration and LR schedules (with sane fallbacks)
        self._entropy_start = float(getattr(cfg.train, 'entropy_coef', 0.02))
        self._entropy_to = float(getattr(cfg.train, 'entropy_anneal_to', max(0.01, self._entropy_start * 0.25)))
        self._entropy_frames = int(getattr(cfg.train, 'entropy_anneal_frames', max(1, self.total_frames)))
        self._eps_start = float(getattr(cfg.train, 'epsilon_greedy', 0.05))
        self._eps_to = float(getattr(cfg.train, 'epsilon_greedy_decay_to', 0.0))
        self._eps_frames = int(getattr(cfg.train, 'epsilon_greedy_decay_frames', max(1, self.total_frames)))
        # LR schedule (linear decay to factor)
        try:
            self._lr0 = float(self.agent.opt.param_groups[0].get('lr', 2e-4))
        except Exception:
            self._lr0 = 2e-4
        self._lr_to = float(getattr(cfg.train, 'learning_rate_decay_to', self._lr0 * 0.5))

        # Threads
        self._collector_thread: Optional[threading.Thread] = None
        self._loader_thread: Optional[threading.Thread] = None

        # TensorBoard (optional)
        self._tb = None
        self._tb_last_flush_t = time.perf_counter()
        try:
            if SummaryWriter is not None:
                log_dir = getattr(cfg, 'log_dir', 'runs/dreamer_crafter')
                self._tb = SummaryWriter(log_dir=log_dir)
        except Exception:
            self._tb = None

        # RND (intrinsic reward) optional
        self._rnd = None
        try:
            if bool(getattr(self.cfg.train, "use_intrinsic", False)):
                ch = int(getattr(self.cfg.model, "obs_channels", 3))
                self._rnd = RND(in_channels=ch).to(self.device)
                self._intrinsic_coef = float(getattr(self.cfg.train, "intrinsic_coef", 0.5))
                self._intrinsic_norm = bool(getattr(self.cfg.train, "intrinsic_norm", True))
                self._intrinsic_update_every = int(getattr(self.cfg.train, "intrinsic_update_every", 4))
                # EMA stats for normalization
                self._ri_mean = 0.0
                self._ri_var = 1.0
        except Exception:
            self._rnd = None

        # PriorNet for LLM distillation (optional)
        self._priornet = None
        self._priornet_opt = None
        self._priornet_last_update = 0
        try:
            if bool(getattr(self.cfg.train, "llm_priornet_enabled", False)):
                latent_dim = int(getattr(self.cfg.model, "latent_dim", 256))
                n_act = int(getattr(self.agent, "n_actions", 0))
                hid = int(getattr(self.cfg.train, "llm_priornet_hidden", 256))
                import torch.nn as nn  # local import to avoid top-level changes
                self._priornet = nn.Sequential(
                    nn.Linear(latent_dim, hid),
                    nn.ELU(),
                    nn.Linear(hid, n_act),
                ).to(self.device)
                import torch
                self._priornet_opt = torch.optim.Adam(self._priornet.parameters(), lr=float(getattr(self.cfg.train, "llm_priornet_lr", 1e-3)))
                self._priornet_temp = float(getattr(self.cfg.train, "llm_priornet_temp", 2.0))
        except Exception:
            self._priornet = None
            self._priornet_opt = None

        # Synthetic experience generator (optional)
        self.synthetic_generator = None
        self.synthetic_generation_interval = int(getattr(cfg.train, 'synthetic_generation_interval', 0))
        self.last_synth_gen_step = 0
        try:
            from ..utils.synthetic_generator import SyntheticExperienceGenerator  # type: ignore
            if (EnhancedReplayBuffer is not None) and use_synth and (self.synthetic_generation_interval > 0) and (self.llm is not None):
                self.synthetic_generator = SyntheticExperienceGenerator(
                    world_model=self.agent.world,
                    llm_adapter=self.llm,
                    device=self.device,
                    max_rollout_length=int(getattr(cfg.train, 'synthetic_rollout_length', 5)),
                    confidence_threshold=float(getattr(cfg.train, 'synthetic_confidence_threshold', 0.3)),
                    success_reward_threshold=float(getattr(cfg.train, 'synthetic_success_threshold', 0.1)),
                    synthetic_execution_prob=float(getattr(cfg.train, 'synthetic_execution_prob', 0.0)),
                )
        except Exception:
            self.synthetic_generator = None

    # -------------- Utility --------------
    def _add_to_replay(self, sample_td, *, is_synthetic: bool = False, synth_meta: Optional[dict] = None) -> None:
        """Add a sample to the replay, routing to EnhancedReplayBuffer if available.

        Accepts a TensorDict-like row (single) and converts to plain dict when needed.
        """
        try:
            # EnhancedReplayBuffer path
            if hasattr(self.replay, 'add_real'):
                # Convert to plain dict expected by EnhancedReplayBuffer
                def _get(td, key):
                    try:
                        return td.get(key)
                    except Exception:
                        return None
                sample_py = {
                    'observation': _get(sample_td, 'observation'),
                    'action': _get(sample_td, 'action'),
                    'reward': _get(sample_td, 'reward'),
                    'done': _get(sample_td, 'done'),
                    'next': {'observation': _get(sample_td, ('next', 'observation'))},
                }
                # Preserve LLM fields if present
                for k in ['llm_prior_logits', 'llm_confidence', 'llm_features']:
                    v = _get(sample_td, k)
                    if v is not None:
                        sample_py[k] = v
                if is_synthetic and synth_meta is not None and hasattr(self.replay, 'add_synthetic'):
                    try:
                        self.replay.add_synthetic(
                            sample=sample_py,
                            advice_id=synth_meta.get('advice_id', 'unknown'),
                            llm_confidence=float(synth_meta.get('llm_confidence', 0.5)),
                            execution_success=bool(synth_meta.get('execution_success', True)),
                            synthetic_plan=synth_meta.get('synthetic_plan'),
                            base_weight=float(synth_meta.get('w_synth', 0.3)),
                        )
                    except Exception:
                        # fallback to add_real on failure
                        self.replay.add_real(sample_py)
                else:
                    self.replay.add_real(sample_py)
                return
        except Exception:
            pass
        # Fallback: standard replay with add()
        try:
            self.replay.add(sample_td)
        except Exception:
            pass
    def _get_obs_tensor(self, td) -> torch.Tensor:
        obs = td.get(self.obs_key)
        if obs is None and ("next", self.obs_key) in td.keys(True, True):
            obs = td.get(("next", self.obs_key))
        if obs.dim() == 3 and obs.shape[-1] in (1, 3) and obs.shape[0] not in (1, 3):
            obs = obs.permute(2, 0, 1).contiguous()
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255.0)
        else:
            obs = obs.float()
        return obs

    def _append_simple(self, prev_td, action, next_td, llm_tuple: Optional[tuple[int, dict]] = None) -> None:
        # Minimal per-sample append on CPU replay
        try:
            from tensordict import TensorDict  # type: ignore
        except Exception:  # pragma: no cover
            from torchrl.data import TensorDict  # type: ignore

        def _get(td_in, key, fb=None):
            v = td_in.get(key)
            if v is None and fb is not None:
                v = td_in.get(fb)
            return v

        prev_obs = _get(prev_td, self.obs_key, ("next", self.obs_key))
        next_obs = _get(next_td, ("next", self.obs_key), self.obs_key)
        rew = _get(next_td, ("next", "reward"), "reward")
        dn = _get(next_td, ("next", "done"), "done")

        def _ensure_tensor(x, default=0.0):
            if x is None:
                return torch.tensor(default)
            return x if isinstance(x, torch.Tensor) else torch.tensor(x)

        prev_obs = _ensure_tensor(prev_obs)
        next_obs = _ensure_tensor(next_obs)
        rew = _ensure_tensor(rew)
        dn = _ensure_tensor(dn, 0.0)
        act = _ensure_tensor(action)

        # If batched, handle vectorized extend
        if isinstance(prev_obs, torch.Tensor) and prev_obs.ndim > 3:
            n = prev_obs.shape[0]

            def _as_col(x: torch.Tensor, n_env: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
                t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
                if dtype is not None:
                    t = t.to(dtype)
                if t.numel() == 1:
                    return t.view(1, 1).repeat(n_env, 1)
                if t.ndim == 1:
                    return t.view(-1, 1)
                if t.ndim >= 2 and t.shape[1] != 1:
                    t = t[:, :1]
                return t

            def _to_uint8_b(x: torch.Tensor) -> torch.Tensor:
                if x.dtype == torch.uint8:
                    return x.cpu()
                return x.clamp(0, 1).mul(255).to(torch.uint8).cpu()

            o_prev_b = _to_uint8_b(prev_obs)
            if isinstance(next_obs, torch.Tensor) and next_obs.ndim > 3:
                o_next_b = _to_uint8_b(next_obs)
            else:
                o_next_b = o_prev_b

            a_b = _as_col(act, n, dtype=torch.long).cpu()
            r_b = _as_col(rew, n, dtype=torch.float32).cpu()
            d_b = _as_col(dn, n, dtype=torch.float32).cpu()

            # Per-row add to allow selective LLM annotations
            for i in range(n):
                row = TensorDict(
                    {
                        "observation": o_prev_b[i],
                        "action": a_b[i:i+1],
                        "reward": r_b[i:i+1],
                        "done": d_b[i:i+1],
                        "next": {"observation": o_next_b[i]},
                    },
                    batch_size=[],
                )
                # If LLM provided for specific index, attach extra fields
                try:
                    if llm_tuple is not None:
                        llm_idx, llm_out = llm_tuple
                        if i == int(llm_idx) and isinstance(llm_out, dict):
                            pl = llm_out.get("prior_logits")
                            cf = llm_out.get("confidence")
                            ft = llm_out.get("features")
                            if pl is not None:
                                row.set("llm_prior_logits", torch.as_tensor(pl).cpu())
                            if cf is not None:
                                row.set("llm_confidence", torch.as_tensor(cf).cpu())
                            if ft is not None:
                                row.set("llm_features", torch.as_tensor(ft).cpu())
                except Exception:
                    pass
                self._add_to_replay(row)
            return

        # Single sample path
        def _to_uint8(x: torch.Tensor) -> torch.Tensor:
            if x.dtype == torch.uint8:
                return x.cpu()
            return x.clamp(0, 1).mul(255).to(torch.uint8).cpu()

        if isinstance(prev_obs, torch.Tensor) and prev_obs.ndim > 3:
            prev_obs = prev_obs[0]
        if isinstance(next_obs, torch.Tensor) and next_obs.ndim > 3:
            next_obs = next_obs[0]
        if isinstance(act, torch.Tensor) and act.ndim > 1:
            act = act[0]

        try:
            from tensordict import TensorDict  # type: ignore
        except Exception:  # pragma: no cover
            from torchrl.data import TensorDict  # type: ignore

        sample_s = TensorDict(
            {
                "observation": _to_uint8(prev_obs) if isinstance(prev_obs, torch.Tensor) else prev_obs,
                "action": (act if isinstance(act, torch.Tensor) else torch.tensor(act)).long().view(1, -1).cpu(),
                "reward": (rew if isinstance(rew, torch.Tensor) else torch.tensor(rew)).to(torch.float32).view(1, -1).cpu(),
                "done": (dn if isinstance(dn, torch.Tensor) else torch.tensor(dn)).to(torch.float32).view(1, -1).cpu(),
                "next": {"observation": _to_uint8(next_obs) if isinstance(next_obs, torch.Tensor) else next_obs},
            },
            batch_size=[],
        )
        # Attach LLM if provided (single sample path)
        try:
            if llm_tuple is not None:
                llm_idx, llm_out = llm_tuple
                if isinstance(llm_out, dict):
                    pl = llm_out.get("prior_logits")
                    cf = llm_out.get("confidence")
                    ft = llm_out.get("features")
                    if pl is not None:
                        sample_s.set("llm_prior_logits", torch.as_tensor(pl).cpu())
                    if cf is not None:
                        sample_s.set("llm_confidence", torch.as_tensor(cf).cpu())
                    if ft is not None:
                        sample_s.set("llm_features", torch.as_tensor(ft).cpu())
        except Exception:
            pass
        self._add_to_replay(sample_s)

    # -------------- Threads --------------
    def _collector_loop(self) -> None:
        td = self._last_td
        local_env = None
        error_count = 0
        while not self._stop["v"]:
            try:
                # lazily create thread-local env if needed
                if (td is None or local_env is None) and make_crafter_env is not None:
                    try:
                        local_env = make_crafter_env(self.cfg.env)
                        td = local_env.reset()
                        # refresh obs_key from local env spec
                        try:
                            self.obs_key = "pixels" if "pixels" in local_env.observation_spec.keys(True, True) else "observation"
                        except Exception:
                            self.obs_key = "observation"
                    except Exception:
                        local_env = None

                env_to_use = local_env if local_env is not None else self.env
                if env_to_use is None or td is None:
                    raise RuntimeError("env not ready in collector")

                obs = self._get_obs_tensor(td)
                with torch.no_grad():
                    # 1) 初期ランダム期間（真の一様乱択）
                    if self._stats.frames_total < max(1, int(getattr(self.cfg.train, 'init_random_frames', 0))):
                        import torch as _torch
                        n = int(getattr(self.agent, 'n_actions', 1))
                        b = int(obs.shape[0]) if obs.dim() >= 4 else 1
                        act = _torch.randint(low=0, high=n, size=(b, 1))
                    # 2) 早期はsoftmax温度サンプリング
                    elif self._stats.frames_total < self.exploration_softmax_frames:
                        # Fast latent for logits
                        z = self.agent._enc_cache(obs.to(self.device))
                        h_seq, _ = self.agent.world.rssm(
                            z.unsqueeze(1), torch.zeros(1, z.size(0), z.size(-1), device=z.device)
                        )
                        h = h_seq.squeeze(1)
                        logits = self.agent.ac.policy_logits(h)
                        import torch.nn.functional as F
                        # Linear anneal temperature: high→low
                        frac = float(self._stats.frames_total - int(getattr(self.cfg.train, 'init_random_frames', 0))) / max(1, (self.exploration_softmax_frames - int(getattr(self.cfg.train, 'init_random_frames', 0))))
                        frac = float(max(0.0, min(1.0, frac)))
                        temp = self.softmax_temp_start + (self.softmax_temp_end - self.softmax_temp_start) * frac
                        temp = max(0.8, float(temp))
                        probs = F.softmax(logits / temp, dim=-1)
                        act = torch.multinomial(probs, num_samples=1).cpu()
                    else:
                        # 後半はargmaxベース。ただしentropyが低すぎるときは低温softmaxを小確率で再導入
                        use_late_softmax = False
                        try:
                            if (self._stats.last_entropy is not None) and (self._stats.last_entropy < 0.02):
                                import torch as _torch
                                use_late_softmax = (_torch.rand(1).item() < self.late_softmax_prob)
                        except Exception:
                            use_late_softmax = False
                        if use_late_softmax:
                            z = self.agent._enc_cache(obs.to(self.device))
                            h_seq, _ = self.agent.world.rssm(
                                z.unsqueeze(1), torch.zeros(1, z.size(0), z.size(-1), device=z.device)
                            )
                            h = h_seq.squeeze(1)
                            logits = self.agent.ac.policy_logits(h)
                            import torch.nn.functional as F
                            probs = F.softmax(logits / float(self.late_softmax_temp), dim=-1)
                            act = torch.multinomial(probs, num_samples=1).cpu()
                        else:
                            act = self.agent.act(obs.to(self.device)).cpu()
                td.set("action", act)
                td = env_to_use.step(td)
                # Reset environments on done (terminated or truncated)
                try:
                    dn = None
                    if ("next", "done") in td.keys(True, True):
                        dn = td.get(("next", "done"))
                    elif "done" in td.keys(True, True):
                        dn = td.get("done")
                    # Handle Gymnasium-style terminated/truncated flags if present in info
                    try:
                        if dn is None and ("next", "info") in td.keys(True, True):
                            inf = td.get(("next", "info"))
                            term = None
                            trunc = None
                            if isinstance(inf, dict):
                                term = inf.get("terminated")
                                trunc = inf.get("truncated")
                            # best-effort coercion
                            if term is not None or trunc is not None:
                                t = term if term is not None else 0
                                c = trunc if trunc is not None else 0
                                dn = (torch.as_tensor(t).to(torch.bool) | torch.as_tensor(c).to(torch.bool)).to(torch.float32)
                    except Exception:
                        pass
                    done_mask = dn.to(dtype=torch.bool).view(-1) if dn is not None else None
                except Exception:
                    done_mask = None

                # Intrinsic reward (RND)
                if self._rnd is not None:
                    try:
                        obs_i = self._get_obs_tensor(td)
                        # Move to device and ensure float in [0,1]
                        obs_i = (obs_i.to(self.device, non_blocking=True))
                        ri = self._rnd.intrinsic_reward(obs_i)  # [B,1] or [B]
                        # Reduce to per-env scalar
                        if isinstance(ri, torch.Tensor):
                            ri = ri.view(-1)
                            ri_mean_now = float(ri.mean().detach().cpu())
                        else:
                            ri_mean_now = float(ri)
                        # EMA normalization if enabled
                        if self._intrinsic_norm:
                            m = 0.99
                            self._ri_mean = m * self._ri_mean + (1 - m) * ri_mean_now
                            self._ri_var = m * self._ri_var + (1 - m) * (ri_mean_now - self._ri_mean) ** 2
                            denom = (self._ri_var ** 0.5) + 1e-6
                            ri_n = (ri - self._ri_mean) / denom if isinstance(ri, torch.Tensor) else (ri_mean_now - self._ri_mean) / denom
                        else:
                            ri_n = ri
                        # Add shaped intrinsic to reward in td
                        try:
                            # Get base reward tensor (broadcast-safe)
                            if ("next", "reward") in td.keys(True, True):
                                r_any = td.get(("next", "reward"))
                                key_set = ("next", "reward")
                            else:
                                r_any = td.get("reward")
                                key_set = "reward"
                            r_t = r_any if isinstance(r_any, torch.Tensor) else torch.as_tensor(r_any)
                            r_t = r_t.to(dtype=torch.float32, device=self.device)
                            # Log external reward mean before shaping
                            try:
                                with self._lock:
                                    self._stats.last_step_reward_ext_mean = float(r_t.view(-1).mean().detach().cpu())
                            except Exception:
                                pass
                            # Align shapes to (B,1)
                            if r_t.dim() == 0:
                                r_t = r_t.view(1, 1)
                            elif r_t.dim() == 1:
                                r_t = r_t.view(-1, 1)
                            if isinstance(ri_n, torch.Tensor):
                                ri_col = ri_n.to(dtype=torch.float32, device=self.device).view(-1, 1)
                                # Broadcast/truncate to match batch
                                n_r = r_t.shape[0]
                                if ri_col.shape[0] != n_r:
                                    if ri_col.numel() == 1:
                                        ri_col = ri_col.view(1, 1).repeat(n_r, 1)
                                    else:
                                        ri_col = ri_col[:n_r]
                            else:
                                ri_col = torch.full_like(r_t, float(ri_n))
                            # Optional: clip intrinsic to [0, ri_clip_max]
                            ri_clip_max = float(getattr(self.cfg.train, 'intrinsic_clip_max', 0.5))
                            if isinstance(ri_col, torch.Tensor):
                                ri_eff = torch.clamp(ri_col, min=0.0, max=ri_clip_max)
                            else:
                                ri_eff = min(ri_clip_max, max(0.0, float(ri_col)))
                            # 外的報酬が負のときは内的の重みを半減
                            neg_mask = (r_t < 0).to(r_t.dtype)
                            alpha = float(self._intrinsic_coef)
                            if isinstance(ri_eff, torch.Tensor):
                                alpha_t = torch.full_like(r_t, alpha)
                                alpha_t = torch.where(neg_mask > 0, alpha_t * 0.5, alpha_t)
                                r_new = r_t + alpha_t * ri_eff
                            else:
                                alpha_s = alpha * (0.5 if (r_t.mean().item() < 0) else 1.0)
                                r_new = r_t + alpha_s * ri_eff
                            # Write back to CPU tensor for replay packing later
                            td.set(key_set, r_new.detach().cpu())
                            # Log intrinsic mean after clamp
                            try:
                                with self._lock:
                                    self._stats.last_step_intrinsic_mean = float((ri_eff if isinstance(ri_eff, torch.Tensor) else torch.tensor(ri_eff)).view(-1).mean().detach().cpu())
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # Train predictor periodically
                        try:
                            if (self._stats.frames_total % max(1, self._intrinsic_update_every)) == 0:
                                _ = self._rnd.update(obs_i)
                        except Exception:
                            pass
                        # TB: intrinsic mean
                        try:
                            if self._tb is not None:
                                step_tb = int(self._stats.frames_total)
                                self._tb.add_scalar('intrinsic/ri_mean', float(ri_mean_now), step_tb)
                                if self._stats.last_step_intrinsic_mean is not None:
                                    self._tb.add_scalar('intrinsic/ri_eff_mean', float(self._stats.last_step_intrinsic_mean), step_tb)
                                if self._stats.last_step_reward_ext_mean is not None:
                                    self._tb.add_scalar('env/step_reward_ext_mean', float(self._stats.last_step_reward_ext_mean), step_tb)
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    # If RND disabled, inject epsilon-greedy random action occasionally in early phase
                    try:
                        if self._stats.frames_total < (self.exploration_softmax_frames // 2):
                            import torch as _torch
                            if _torch.rand(1).item() < 0.05:
                                n = int(getattr(self.agent, 'n_actions', 1))
                                act = _torch.randint(low=0, high=n, size=(act.shape[0], 1))
                                td.set('action', act)
                    except Exception:
                        pass
                # Episode accumulators update
                try:
                    rew_vec = td.get(("next", "reward")) if ("next", "reward") in td.keys(True, True) else td.get("reward")
                    if rew_vec is not None:
                        rv = rew_vec.view(-1).to(dtype=torch.float32)
                        # step平均報酬を記録（TBにも出す）
                        step_mean = float(rv.mean().detach().cpu()) if rv.numel() > 0 else 0.0
                        with self._lock:
                            self._stats.last_step_reward_mean = step_mean
                        if self._tb is not None:
                            self._tb.add_scalar('env/step_reward_mean', step_mean, int(self._stats.frames_total))
                            # RND付与後の報酬かどうかはログ名で区別も検討可
                    # 外的報酬（RND加算前）も併せて保持
                    try:
                        r_any = None
                        if ("next", "reward") in td.keys(True, True):
                            r_any = td.get(("next", "reward"))
                        elif "reward" in td.keys(True, True):
                            r_any = td.get("reward")
                        if r_any is not None:
                            r_ext = torch.as_tensor(r_any).view(-1).to(torch.float32)
                            if self._tb is not None:
                                self._tb.add_scalar('env/step_reward_ext_mean_observed', float(r_ext.mean().detach().cpu()), int(self._stats.frames_total))
                    except Exception:
                        pass
                    if rew_vec is not None and self._ep_ret_env is not None:
                        rv = rew_vec.view(-1).to(dtype=torch.float32)
                        ncap = min(rv.numel(), self._ep_ret_env.numel())
                        self._ep_ret_env[:ncap] += rv[:ncap]
                        # 外的報酬（観測時点のr_anyを別に積算）
                        try:
                            if self._ep_ret_ext_env is not None:
                                r_any = None
                                if ("next", "reward") in td.keys(True, True):
                                    r_any = td.get(("next", "reward"))
                                elif "reward" in td.keys(True, True):
                                    r_any = td.get("reward")
                                if r_any is not None:
                                    r_ext = torch.as_tensor(r_any).view(-1).to(torch.float32)
                                    self._ep_ret_ext_env[:ncap] += r_ext[:ncap]
                        except Exception:
                            pass
                        if self._ep_len_env is not None:
                            self._ep_len_env[:ncap] += 1
                except Exception:
                    pass
                if done_mask is not None and done_mask.any():
                    try:
                        td_reset = None
                        try:
                            idx = torch.nonzero(done_mask, as_tuple=False).view(-1).cpu().tolist()
                            td_reset = env_to_use.reset(env_ids=idx)  # type: ignore
                        except Exception:
                            td_reset = env_to_use.reset()
                        if td_reset is not None:
                            td = td_reset
                    except Exception:
                        pass
                    # Per-episode metrics (Score%% and Return)
                    try:
                        idxs = torch.nonzero(done_mask, as_tuple=False).view(-1).cpu().tolist()
                        for _i in idxs:
                            # Episode return
                            ret_i = None
                            if self._ep_ret_env is not None and 0 <= _i < self._ep_ret_env.numel():
                                ret_i = float(self._ep_ret_env[_i].item())
                            # Extract achievements from info
                            ach = None
                            try:
                                info_any = None
                                if ("next", "info") in td.keys(True, True):
                                    info_any = td.get(("next", "info"))
                                elif "info" in td.keys(True, True):
                                    info_any = td.get("info")
                                if isinstance(info_any, (list, tuple)) and 0 <= _i < len(info_any) and isinstance(info_any[_i], dict):
                                    ach = info_any[_i].get("achievements") or info_any[_i].get("achievements/boolean")
                                elif isinstance(info_any, dict):
                                    ach = info_any.get("achievements") or info_any.get("achievements/boolean")
                                else:
                                    try:
                                        from tensordict import TensorDict as _TD  # type: ignore
                                        if isinstance(info_any, _TD):
                                            ach = {k: v for k, v in info_any.items() if isinstance(v, (int, float, bool))}
                                    except Exception:
                                        pass
                            except Exception:
                                ach = None
                            score_percent = None
                            if isinstance(ach, dict) and len(ach) > 0:
                                total_keys = 22
                                achieved = sum(1 for v in ach.values() if bool(v))
                                score_percent = (achieved / float(total_keys)) * 100.0
                            # Update rolling stats
                            with self._lock:
                                if ret_i is not None:
                                    self._stats.episode_returns.append(ret_i)
                                    if len(self._stats.episode_returns) > 100:
                                        self._stats.episode_returns.pop(0)
                                    # 外的報酬のエピソード合計
                                    try:
                                        if self._ep_ret_ext_env is not None and 0 <= _i < self._ep_ret_ext_env.numel():
                                            self._stats.episode_returns_ext.append(float(self._ep_ret_ext_env[_i].item()))
                                            if len(self._stats.episode_returns_ext) > 100:
                                                self._stats.episode_returns_ext.pop(0)
                                    except Exception:
                                        pass
                                if score_percent is not None:
                                    self._stats.score_percents.append(float(score_percent))
                                    if len(self._stats.score_percents) > 100:
                                        self._stats.score_percents.pop(0)
                                self._stats.episodes_total += 1
                                step_tb = int(self._stats.frames_total)
                            # Emit TB scalars at episode boundary
                            try:
                                if self._tb is not None:
                                    if ret_i is not None:
                                        self._tb.add_scalar('episode/return', float(ret_i), step_tb)
                                    if score_percent is not None:
                                        self._tb.add_scalar('episode/score_percent', float(score_percent), step_tb)
                            except Exception:
                                pass
                            # Reset per-env accumulators
                            if self._ep_ret_env is not None and 0 <= _i < self._ep_ret_env.numel():
                                self._ep_ret_env[_i] = 0.0
                            if self._ep_ret_ext_env is not None and 0 <= _i < self._ep_ret_ext_env.numel():
                                self._ep_ret_ext_env[_i] = 0.0
                            if self._ep_len_env is not None and 0 <= _i < self._ep_len_env.numel():
                                self._ep_len_env[_i] = 0
                    except Exception:
                        pass
                # Optional LLM call (select one env by beta & cooldown)
                llm_tuple = None
                try:
                    if self._llm_enabled and (self.llm is not None) and (self._llm_call_budget > 0):
                        with torch.no_grad():
                            z = self.agent._enc_cache(obs.to(self.device))
                            h_seq, _ = self.agent.world.rssm(
                                z.unsqueeze(1), torch.zeros(1, z.size(0), z.size(-1), device=z.device)
                            )
                            h = h_seq.squeeze(1)
                            logits_wm, uncertainty = self.agent.ac.policy_logits(h, return_uncertainty=True)
                            beta = self.agent.uncertainty_gate.compute_beta(uncertainty)
                        # Choose candidate env by beta and cooldown
                        import torch as _torch
                        cooldown_ok = (self._llm_steps_since_call_env >= int(self._llm_cooldown))
                        beta_ok = (beta >= float(self._llm_beta_thr))
                        candidates = (cooldown_ok & beta_ok)
                        if bool(candidates.any()):
                            idx = int(_torch.argmax(beta.masked_fill(~candidates, -1e9)).item())
                            # Minimal context
                            ctx = {"items": [{"summary": {"step": int(self._stats.frames_total), "beta": float(beta[idx].item())}}]}
                            # Run LLM
                            obs_i = obs[idx:idx+1].detach().cpu().numpy() if obs.dim() >= 4 else obs.detach().cpu().numpy()
                            out = None
                            try:
                                out = self.llm.infer(obs_i, num_actions=int(self.agent.n_actions), context=ctx)
                            except Exception:
                                out = None
                            if out is not None and isinstance(out, dict) and (out.get("prior_logits") is not None):
                                self._llm_call_budget -= 1
                                self._llm_steps_since_call_env[idx] = 0
                                llm_tuple = (idx, out)
                except Exception:
                    llm_tuple = None

                self._append_simple(self._last_td, act, td, llm_tuple)  # type: ignore[arg-type]
                self._last_td = td
                # Frames accounting
                try:
                    # Prefer batch size from observation to be robust during warmup
                    if ("next", self.obs_key) in td.keys(True, True):
                        ob_next = td.get(("next", self.obs_key))
                    else:
                        ob_next = td.get(self.obs_key)
                    if ob_next is not None and isinstance(ob_next, torch.Tensor):
                        if ob_next.ndim >= 4:
                            env_batch = int(ob_next.shape[0])
                        else:
                            env_batch = 1
                    else:
                        rew_vec = td.get(("next", "reward")) if ("next", "reward") in td.keys(True, True) else td.get("reward")
                        env_batch = int(rew_vec.view(-1).numel()) if rew_vec is not None else 1
                except Exception:
                    env_batch = 1
                with self._lock:
                    self._stats.frames_total += env_batch
                    try:
                        self._stats.replay_size = int(len(self.replay))
                    except Exception:
                        pass
                if self._stats.frames_total >= self.total_frames:
                    break
                # Advance LLM cooldown counters
                try:
                    import torch as _torch
                    self._llm_steps_since_call_env = self._llm_steps_since_call_env + _torch.ones_like(self._llm_steps_since_call_env)
                except Exception:
                    pass

                # Synthetic generation (interval-based)
                try:
                    if self.synthetic_generator is not None and (self._stats.frames_total - self.last_synth_gen_step) >= self.synthetic_generation_interval:
                        # Use current observation as seed
                        cur_obs = self._get_obs_tensor(td).to(self.device)
                        # Simple context and triggers
                        ctx = {"items": [{"summary": {"step": int(self._stats.frames_total)}}]}
                        recent_rew = 0.0
                        td_error = float(self._stats.last_td_abs or 0.0)
                        syn_list = self.synthetic_generator.generate_synthetic_experience(cur_obs, ctx, int(self.agent.n_actions), recent_reward=recent_rew, td_error=td_error)
                        # Append synthetic to EnhancedReplayBuffer if any
                        if syn_list and hasattr(self.replay, 'add_synthetic'):
                            for transition, meta in syn_list:
                                try:
                                    self.replay.add_synthetic(
                                        sample=transition,
                                        advice_id=meta.advice_id,
                                        llm_confidence=meta.llm_confidence,
                                        execution_success=meta.execution_success,
                                        synthetic_plan=meta.synthetic_plan,
                                        base_weight=meta.w_synth,
                                    )
                                except Exception:
                                    pass
                            self.last_synth_gen_step = int(self._stats.frames_total)
                except Exception:
                    pass
                # Advance LLM cooldown counters
                try:
                    import torch as _torch
                    self._llm_steps_since_call_env = self._llm_steps_since_call_env + _torch.ones_like(self._llm_steps_since_call_env)
                except Exception:
                    pass
            except Exception as e:
                error_count += 1
                if error_count in (1, 5, 20):
                    try:
                        print(f"[ASYNC] collector error #{error_count}: {str(e)[:160]}")
                    except Exception:
                        pass
                time.sleep(0.001)
                continue

    def _loader_loop(self) -> None:
        # Preload mini-batches to GPU, double-buffered
        while not self._stop["v"]:
            try:
                # Start as soon as one batch is possible (avoid long cold start)
                if len(self.replay) < self.batch_size:
                    time.sleep(0.001)
                    continue
                batch = self.replay.sample(self.batch_size)
                # Decode uint8 observations to float on GPU
                if hasattr(batch, "pin_memory"):
                    batch = batch.pin_memory()
                batch = batch.to(self.device, non_blocking=True)
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
                with self._batch_lock:
                    # If full, drop the oldest (deque semantics)
                    self._batch_queue.append(batch)
            except Exception:
                time.sleep(0.001)
                continue

    # -------------- Public API --------------
    def train(self) -> None:
        # Spawn threads
        self._collector_thread = threading.Thread(target=self._collector_loop, daemon=True)
        self._loader_thread = threading.Thread(target=self._loader_loop, daemon=True)
        self._collector_thread.start()
        self._loader_thread.start()

        # Learner loop
        t0 = time.perf_counter()
        last_log_t = t0
        last_frames = 0
        last_updates = 0

        try:
            while True:
                # Stop condition
                with self._lock:
                    total_frames = self._stats.frames_total
                if total_frames >= self.total_frames:
                    break

                # Pop a preloaded batch if available
                batch = None
                with self._batch_lock:
                    if self._batch_queue:
                        batch = self._batch_queue.popleft()

                if batch is None:
                    # Heartbeat even when waiting for first batch
                    now = time.perf_counter()
                    if now - last_log_t >= 1.0:
                        with self._lock:
                            f = self._stats.frames_total
                            rep = self._stats.replay_size
                        fps = (f - last_frames) / (now - last_log_t) if now > last_log_t else 0.0
                        print(f"[ASYNC] warming-up replay={rep} frames={f}/{self.total_frames} fps={fps:.1f}")
                        last_log_t = now
                        last_frames = f
                    time.sleep(0.0005)
                    continue

                # Guard: wait for minimum replay init before updates
                with self._lock:
                    cur_frames = int(self._stats.frames_total)
                if cur_frames < max(20000, self.min_replay_init):
                    time.sleep(0.001)
                    continue

                # Update
                t_upd0 = time.perf_counter()
                out: dict[str, Any] = self.agent.update(batch)
                t_upd1 = time.perf_counter()
                dur_ms = (t_upd1 - t_upd0) * 1000.0

                with self._lock:
                    self._stats.updates_total += 1
                    if isinstance(out, dict):
                        # common loss key fallback
                        val = None
                        for k in ("loss", "value/td_abs_mean", "loss/model_recon", "loss/value"):
                            if k in out:
                                try:
                                    val = float(out[k])
                                    break
                                except Exception:
                                    pass
                        self._stats.last_update_loss = val
                    self._stats.last_update_ms = float(dur_ms)
                    # Learning indicators snapshot
                    if isinstance(out, dict):
                        self._stats.last_entropy = float(out.get("policy/entropy", self._stats.last_entropy or 0.0))
                        self._stats.last_value_ev = float(out.get("value/explained_variance", self._stats.last_value_ev or 0.0))
                        self._stats.last_td_abs = float(out.get("value/td_abs_mean", self._stats.last_td_abs or 0.0))
                        self._stats.last_psnr = float(out.get("world/psnr_db", self._stats.last_psnr or 0.0))
                        self._stats.last_grad_norm = float(out.get("optim/grad_global_norm", self._stats.last_grad_norm or 0.0))
                        # Emit TB scalars for training metrics
                        try:
                            if self._tb is not None:
                                step_tb = int(self._stats.frames_total)
                                if self._stats.last_update_loss is not None:
                                    self._tb.add_scalar('train/loss', float(self._stats.last_update_loss), step_tb)
                                self._tb.add_scalar('train/policy_entropy', float(self._stats.last_entropy or 0.0), step_tb)
                                self._tb.add_scalar('train/value_explained_variance', float(self._stats.last_value_ev or 0.0), step_tb)
                                self._tb.add_scalar('train/td_abs_mean', float(self._stats.last_td_abs or 0.0), step_tb)
                                self._tb.add_scalar('world/psnr_db', float(self._stats.last_psnr or 0.0), step_tb)
                                self._tb.add_scalar('optim/grad_global_norm', float(self._stats.last_grad_norm or 0.0), step_tb)
                        except Exception:
                            pass

                # Online schedules (entropy/epsilon/LR)
                try:
                    frac_e = min(1.0, self._stats.frames_total / max(1, self._entropy_frames))
                    ent_now = self._entropy_start + (self._entropy_to - self._entropy_start) * frac_e
                    if hasattr(self.agent, 'entropy_coef'):
                        self.agent.entropy_coef = float(ent_now)
                    # 内的係数をフレームで逓減（線形 → 0）
                    try:
                        coeff0 = float(getattr(self.cfg.train, 'intrinsic_coef', getattr(self, '_intrinsic_coef', 0.5)))
                        T = float(getattr(self.cfg.train, 'intrinsic_anneal_frames', self.total_frames))
                        frac_i = min(1.0, self._stats.frames_total / max(1.0, T))
                        self._intrinsic_coef = max(0.0, coeff0 * (1.0 - frac_i))
                        if self._tb is not None:
                            self._tb.add_scalar('sched/intrinsic_coef', float(self._intrinsic_coef), int(self._stats.frames_total))
                    except Exception:
                        pass
                    frac_eps = min(1.0, self._stats.frames_total / max(1, self._eps_frames))
                    eps_now = self._eps_start + (self._eps_to - self._eps_start) * frac_eps
                    if hasattr(self.agent, 'epsilon_greedy'):
                        self.agent.epsilon_greedy = float(eps_now)
                    # LR linear decay
                    lr_frac = min(1.0, self._stats.frames_total / max(1, self.total_frames))
                    lr_now = self._lr0 + (self._lr_to - self._lr0) * lr_frac
                    try:
                        for g in self.agent.opt.param_groups:
                            g['lr'] = float(lr_now)
                    except Exception:
                        pass
                    # TB: schedules
                    if self._tb is not None:
                        self._tb.add_scalar('sched/entropy_coef', float(ent_now), int(self._stats.frames_total))
                        self._tb.add_scalar('sched/epsilon_greedy', float(eps_now), int(self._stats.frames_total))
                        self._tb.add_scalar('sched/learning_rate', float(lr_now), int(self._stats.frames_total))
                except Exception:
                    pass

                # PriorNet distillation (optional, on current batch when available)
                try:
                    if (self._priornet is not None) and (self._priornet_opt is not None):
                        self._priornet_last_update += 1
                        upd_every = int(getattr(self.cfg.train, "llm_priornet_update_every", 50))
                        if (self._priornet_last_update % upd_every) == 0:
                            import torch
                            import torch.nn.functional as F
                            # Prepare targets from batch if provided
                            has_targets = ("llm_prior_logits" in batch.keys(True, True))
                            if has_targets:
                                obs_t = batch.get("observation")
                                if obs_t.dtype == torch.uint8:
                                    obs_t = obs_t.to(torch.float32).div(255.0)
                                # Compute current latent h with world model encoder+rssm
                                with torch.no_grad():
                                    z = self.agent._enc_cache(obs_t)
                                    h_seq, _ = self.agent.world.rssm(
                                        z.unsqueeze(1), torch.zeros(1, z.size(0), z.size(-1), device=z.device)
                                    )
                                    h_cur = h_seq.squeeze(1)
                                logits_pred = self._priornet(h_cur)
                                targets = batch.get("llm_prior_logits").to(self.device)
                                temp = float(getattr(self, "_priornet_temp", 2.0))
                                p = F.log_softmax(logits_pred / temp, dim=-1)
                                q = F.softmax(targets / temp, dim=-1)
                                mask_nonzero = (targets.abs().sum(dim=1) > 0).float().unsqueeze(1)
                                loss_pn = -(q * p).sum(dim=-1)
                                loss_pn = (loss_pn * mask_nonzero.squeeze(1)).sum() / (mask_nonzero.sum() + 1e-6)
                                self._priornet_opt.zero_grad(set_to_none=True)
                                loss_pn.backward()
                                self._priornet_opt.step()
                                # TB log
                                try:
                                    if self._tb is not None:
                                        step_tb = int(self._stats.frames_total)
                                        self._tb.add_scalar("llm/priornet_distill_loss", float(loss_pn.detach().cpu()), step_tb)
                                except Exception:
                                    pass
                except Exception:
                    pass

                # Periodic status print (1s)
                now = time.perf_counter()
                if now - last_log_t >= 1.0:
                    with self._lock:
                        f = self._stats.frames_total
                        u = self._stats.updates_total
                        rep = self._stats.replay_size
                        loss_val = self._stats.last_update_loss
                        upd_ms = self._stats.last_update_ms
                    dt = now - last_log_t
                    fps = (f - last_frames) / dt if dt > 0 else 0.0
                    ups = (u - last_updates) / dt if dt > 0 else 0.0
                    qfill = len(self._batch_queue)
                    # Rolling averages
                    try:
                        score_avg = sum(self._stats.score_percents[-50:]) / max(1, len(self._stats.score_percents[-50:]))
                    except Exception:
                        score_avg = 0.0
                    try:
                        reward_avg = sum(self._stats.episode_returns[-50:]) / max(1, len(self._stats.episode_returns[-50:]))
                    except Exception:
                        reward_avg = 0.0
                    try:
                        reward_ext_avg = sum(self._stats.episode_returns_ext[-50:]) / max(1, len(self._stats.episode_returns_ext[-50:]))
                    except Exception:
                        reward_ext_avg = 0.0
                    # Compose status with requested metrics and learning indicators
                    msg = (
                        f"[ASYNC] frames={f}/{self.total_frames} fps={fps:.1f} "
                        f"updates={u} ups={ups:.1f} queue={qfill}/2 replay={rep} "
                        f"Score%={score_avg:.1f} Reward={reward_avg:.3f} Reward_ext={reward_ext_avg:.3f} "
                        f"stepR={(self._stats.last_step_reward_mean if self._stats.last_step_reward_mean is not None else 'n/a')} "
                        f"extR={(self._stats.last_step_reward_ext_mean if self._stats.last_step_reward_ext_mean is not None else 'n/a')} "
                        f"ri={(self._stats.last_step_intrinsic_mean if self._stats.last_step_intrinsic_mean is not None else 'n/a')} "
                        f"upd_ms={(upd_ms if upd_ms is not None else 'n/a')} loss={loss_val if loss_val is not None else 'n/a'} "
                        f"entropy={(self._stats.last_entropy if self._stats.last_entropy is not None else 'n/a')} "
                        f"ev={(self._stats.last_value_ev if self._stats.last_value_ev is not None else 'n/a')} "
                        f"tdabs={(self._stats.last_td_abs if self._stats.last_td_abs is not None else 'n/a')} "
                        f"psnr={(self._stats.last_psnr if self._stats.last_psnr is not None else 'n/a')} "
                        f"gnorm={(self._stats.last_grad_norm if self._stats.last_grad_norm is not None else 'n/a')}"
                    )
                    print(msg)
                    # TB: system/aggregates
                    try:
                        if self._tb is not None:
                            self._tb.add_scalar('system/fps', float(fps), f)
                            self._tb.add_scalar('system/ups', float(ups), f)
                            self._tb.add_scalar('system/replay_size', float(rep), f)
                            self._tb.add_scalar('eval/score_percent_avg50', float(score_avg), f)
                            self._tb.add_scalar('eval/episode_return_avg50', float(reward_avg), f)
                            # rate-limited flush
                            now_tb = time.perf_counter()
                            if now_tb - self._tb_last_flush_t >= 5.0:
                                self._tb.flush()
                                self._tb_last_flush_t = now_tb
                    except Exception:
                        pass
                    last_log_t = now
                    last_frames = f
                    last_updates = u

        finally:
            # Stop threads
            self._stop["v"] = True
            try:
                if self._collector_thread is not None:
                    self._collector_thread.join(timeout=0.5)
            except Exception:
                pass
            try:
                if self._loader_thread is not None:
                    self._loader_thread.join(timeout=0.5)
            except Exception:
                pass

            # Final status
            with self._lock:
                f = self._stats.frames_total
                u = self._stats.updates_total
            elapsed = time.perf_counter() - t0
            print(f"[ASYNC] done frames={f} updates={u} elapsed={elapsed:.1f}s")
            try:
                if self._tb is not None:
                    self._tb.flush()
                    self._tb.close()
            except Exception:
                pass


