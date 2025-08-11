from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from ..utils.replay import make_replay
from ..utils.llm_adapter import LLMAdapter, LLMAdapterConfig
from ..utils.rnd import RND


class Trainer:
    def __init__(self, env, agent, logger, cfg, device) -> None:
        self.env = env
        self.agent = agent
        self.logger = logger
        self.cfg = cfg
        self.device = device

        # Replay buffer (TorchRL TensorDict-based)
        self.replay = make_replay(capacity=cfg.train.replay_capacity)

        # Specs
        try:
            self.obs_key = "pixels" if "pixels" in env.observation_spec.keys(True, True) else "observation"
        except Exception:
            self.obs_key = "observation"

        # Reset environment (Gymnasium API via TorchRL wrapper)
        td = self.env.reset()
        # TorchRL GymWrapper returns TensorDict with key "observation" by default
        self._last_td = td

        self.global_step = 0

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
        # Build tensordict samples; supports vector env by iterating env dimension if present
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
        for i in range(n):
            o_prev = prev_obs[i] if isinstance(prev_obs, torch.Tensor) and prev_obs.ndim > 3 else prev_obs
            o_next = next_obs[i] if isinstance(next_obs, torch.Tensor) and next_obs.ndim > 3 else next_obs
            a_i = act[i] if isinstance(act, torch.Tensor) and act.ndim > 1 else act
            r_i = rew[i] if isinstance(rew, torch.Tensor) and rew.ndim > 0 else rew
            d_i = done[i] if isinstance(done, torch.Tensor) and done.ndim > 0 else done

        sample = TensorDict(
            {
                    "observation": o_prev.cpu(),
                    "action": a_i.cpu(),
                    "reward": r_i.cpu(),
                    "done": d_i.cpu(),
                    "next": {"observation": o_next.cpu()},
            },
            batch_size=[],
        )

        # Optional LLM annotations, per-env slice if provided as batch
        if llm_out is not None:
            import torch as _torch
            pl = llm_out.get("prior_logits") if isinstance(llm_out, dict) else None
            cf = llm_out.get("confidence") if isinstance(llm_out, dict) else None
            ft = llm_out.get("features") if isinstance(llm_out, dict) else None
            if pl is not None:
                pli = pl[i] if isinstance(pl, torch.Tensor) and pl.dim() >= 2 else _torch.tensor(pl)
                sample.set("llm_prior_logits", pli.cpu())
            if cf is not None:
                cfi = cf[i] if isinstance(cf, torch.Tensor) and cf.dim() >= 2 else _torch.tensor(cf)
                sample.set("llm_confidence", cfi.cpu())
            if ft is not None and getattr(self.cfg.model, "llm_features_dim", 0) > 0:
                fti = ft[i] if isinstance(ft, torch.Tensor) and ft.dim() >= 2 else _torch.tensor(ft)
                sample.set("llm_features", fti.cpu())

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
            try:
                if getattr(self.cfg.train, "use_llm", False):
                    # Compute uncertainty & beta quickly from current policy (batch)
                    with torch.no_grad():
                        z = self.agent._enc_cache(obs)
                        h, _ = self.agent.world.rssm(
                            z.unsqueeze(1), torch.zeros(1, z.size(0), z.size(-1), device=z.device)
                        )
                        h = h.squeeze(1)  # [N,latent]
                        base_logits, uncertainty = self.agent.ac.policy_logits(
                            h, return_uncertainty=True
                        )
                        beta = self.agent.uncertainty_gate.compute_beta(uncertainty)  # [N]

                    import numpy as _np
                    n = h.size(0)
                    num_actions = int(getattr(self.agent, "n_actions", 0))
                    # Prepare per-env outputs (default zeros)
                    prior_logits_pack = torch.zeros(n, num_actions, dtype=torch.float32)
                    features_dim = int(getattr(self.cfg.model, "llm_features_dim", 0))
                    features_pack = torch.zeros(n, features_dim, dtype=torch.float32) if features_dim > 0 else None
                    conf_pack = torch.zeros(n, 1, dtype=torch.float32)

                    # Build optional context per env (latent / summary / image) with debug logs
                    context = None
                    try:
                        import os as _os
                        _dbg_on = _os.environ.get("LORE_DEBUG_LLM", "0") in ("1", "true", "True")
                        send_latent = bool(getattr(self.cfg.train, "llm_send_latent", True))
                        send_summary = bool(getattr(self.cfg.train, "llm_send_summary", True))
                        send_image = bool(getattr(self.cfg.train, "llm_send_image", True))
                        latent_dim = int(getattr(self.cfg.train, "llm_latent_dim", 32))
                        img_s = int(getattr(self.cfg.train, "llm_image_size", 16))
                        single_ch = bool(getattr(self.cfg.train, "llm_image_single_channel", True))
                        context = {"items": []}
                        # latent projector (single shared) built on-demand
                        if send_latent and not hasattr(self, "_llm_latent_proj"):
                            in_d = h.size(-1)
                            self._llm_latent_proj = torch.nn.Linear(in_d, latent_dim).to(self.device).eval()
                        for i in range(n):
                            item = {}
                            if send_latent and hasattr(self, "_llm_latent_proj"):
                                with torch.no_grad():
                                    lp = self._llm_latent_proj(h[i:i+1]).squeeze(0).detach().cpu().float()
                                item["latent"] = [float(v) for v in lp[:latent_dim]]
                            if send_summary:
                                item["summary"] = {
                                    "step": int(self.global_step),
                                    "no_reward_steps": int(self._no_reward_steps[i].item()) if i < self._no_reward_steps.numel() else 0,
                                    "beta": float(beta[i].detach().cpu().item()) if i < beta.numel() else 0.0,
                                }
                            if send_image:
                                try:
                                    # downsample obs[i]: [C,H,W] expected
                                    oi = obs[i] if obs.dim() >= 4 else obs
                                    oi = torch.nn.functional.interpolate(oi.unsqueeze(0), size=(img_s, img_s), mode="area").squeeze(0)
                                    if single_ch and oi.dim() == 3 and oi.shape[0] > 1:
                                        oi = oi[:1]
                                    oi = oi.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
                                    item["image"] = oi.tolist()  # small size, keep as nested list
                                except Exception:
                                    pass
                            context["items"].append(item)
                        if _dbg_on:
                            try:
                                print(f"[Trainer] LLM context built items={len(context['items'])} latent_dim={latent_dim} img={img_s}x{img_s}")
                            except Exception:
                                pass
                    except Exception:
                        context = None

                    # Build cache keys from latent per env
                    cache_keys = [None] * n
                    if self._llm_use_cache:
                        for i in range(n):
                            try:
                                v = h[i].detach().float()
                                v = v / (v.norm() + 1e-8)
                                q = (v * 32.0).clamp(-127, 127).to(torch.int8)
                                cache_keys[i] = bytes(q.cpu().numpy().tobytes())
                            except Exception:
                                cache_keys[i] = None

                    # Try cache for each env
                    cached_mask = torch.zeros(n, dtype=torch.bool)
                    for i in range(n):
                        ck = cache_keys[i]
                        if ck is not None and ck in self._llm_cache:
                            out = self._llm_cache[ck]
                            logits = torch.as_tensor(out.get("prior_logits"), dtype=torch.float32)
                            if logits.numel() == num_actions:
                                prior_logits_pack[i] = logits
                                cached_mask[i] = True
                                self._llm_cache_hits += 1
                            else:
                                self._llm_cache_misses += 1

                    # PriorNet fill for non-cached rows (usage rate計測)
                    if self._priornet is not None:
                        try:
                            with torch.no_grad():
                                pn_logits = self._priornet(h.detach())  # [N,A]
                            empty_rows = (prior_logits_pack.abs().sum(dim=1) == 0)
                            # opportunities: rows that were empty after cache
                            self._priornet_opportunities_total += int(empty_rows.sum().item())
                            prior_logits_pack[empty_rows] = pn_logits[empty_rows].detach().cpu()
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
                        beta_cpu = beta.detach().cpu()
                        beta_ok = (beta_cpu >= beta_threshold)
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
                            # Pick highest beta among candidates
                            beta_sel = beta_cpu.masked_fill(~candidates, -1e9)
                            idx = int(torch.argmax(beta_sel).item())
                            try:
                                obs_i = obs[idx:idx+1].detach().cpu().numpy()
                                ctx_i = None
                                try:
                                    if context and isinstance(context, dict) and "items" in context and len(context["items"]) > idx:
                                        ctx_i = {"items": [context["items"][idx]]}
                                except Exception:
                                    ctx_i = context
                                out = self.llm.infer(obs_i, num_actions=num_actions, context=ctx_i)
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
                                    self._llm_beta_when_call.append(float(beta[idx].detach().cpu().item()))
                                    self._llm_calls_total += 1
                                    self._llm_call_budget -= 1
                                    self._llm_steps_since_call_env[idx] = 0
                                    self._llm_calls_env[idx] += 1
                                    # Cache store
                                    ck = cache_keys[idx]
                                    if ck is not None:
                                        self._llm_cache[ck] = out
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
                        "_context": context,
                    }
            except Exception:
                llm_out = None

            # 初期ランダム行動期間
            with torch.no_grad():
                if self.global_step < self._random_warmup_steps:
                    import torch as _torch
                    n = int(getattr(self.agent, "n_actions", 1))
                    batch_n = int(obs.shape[0]) if obs.dim() >= 4 else 1
                    action = _torch.randint(low=0, high=n, size=(batch_n, 1), device=self.device)
                else:
                    action = self.agent.act(obs)
            # TorchRL envs expect action within a tensordict
            td.set("action", action.cpu())
            td = self.env.step(td)

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
                    # Train predictor a bit
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
                            self.logger.add_scalar("env/episode_return", ret_i, self.global_step)
                        if self._ep_success_env is not None:
                            s_i = 1 if bool(self._ep_success_env[_i].item()) else 0
                            self._success_hist.append(s_i)
                            if len(self._success_hist) > self._success_hist_cap:
                                self._success_hist.pop(0)
                            rate = sum(self._success_hist) / max(1, len(self._success_hist))
                            self.logger.add_scalar("env/success_rate", float(rate), self.global_step)
                        # Crafter achievements in info (if provided by env)
                        try:
                            # Try various locations/types for info
                            info_any = None
                            if ("next", "info") in td.keys(True, True):
                                info_any = td.get(("next", "info"))
                            elif "info" in td.keys(True, True):
                                info_any = td.get("info")
                            # Normalize to list[dict] per env if possible
                            infos_list = []
                            if isinstance(info_any, dict):
                                infos_list = [info_any]
                            else:
                                try:
                                    # TensorDict with per-env dicts
                                    from tensordict import TensorDict as _TD  # type: ignore
                                    if isinstance(info_any, _TD):
                                        # best-effort conversion
                                        infos_list = [ {k: v for k, v in info_any.items()} ]
                                except Exception:
                                    pass
                                if not infos_list and isinstance(info_any, (list, tuple)):
                                    infos_list = [x for x in info_any if isinstance(x, dict)]
                            # Aggregate achievements if present
                            if infos_list:
                                ach_any: dict[str, bool] = {}
                                for d in infos_list:
                                    ach = d.get("achievements") or d.get("achievements/boolean") or d.get("achievements_bool")
                                    if isinstance(ach, dict):
                                        for k, v in ach.items():
                                            ach_any[k] = bool(v) or ach_any.get(k, False)
                                if ach_any:
                                    for k, v in ach_any.items():
                                        if v:
                                            self._ach_counts[k] = self._ach_counts.get(k, 0) + 1
                                    self._ach_episodes += 1
                                    total_keys = len(self._ach_counts)
                                    if total_keys > 0 and self._ach_episodes > 0:
                                        mean_success = sum(self._ach_counts.values()) / float(self._ach_episodes * total_keys)
                                        self.logger.add_scalar("env/success_rate", float(mean_success), self.global_step)
                                        import math as _math
                                        gm = 1.0
                                        for c in self._ach_counts.values():
                                            p = max(1e-6, c / float(self._ach_episodes))
                                            gm *= p
                                        crafter_score = gm ** (1.0 / total_keys)
                                        self.logger.add_scalar("env/crafter_score", float(crafter_score), self.global_step)
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

            self._append_to_replay(self._last_td, action, td, llm_out)
            self._last_td = td

            self.global_step += 1
            # Update per-env trackers
            try:
                rew_vec = td.get(("next", "reward")) if ("next", "reward") in td.keys(True, True) else td.get("reward")
                if rew_vec is not None:
                    rew_vec = rew_vec.view(-1)
                    self._no_reward_steps = torch.where(rew_vec != 0.0, torch.zeros_like(self._no_reward_steps), self._no_reward_steps + 1)
            except Exception:
                pass
            self._llm_steps_since_call_env += 1

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
        for _ in range(updates):
            try:
                batch = self.replay.sample(self.cfg.train.batch_size)
            except Exception:
                break  # not enough samples yet
            # move to device
            for k in ["observation", "action", "reward"]:
                if k in batch.keys(True, True):
                    batch.set(k, batch.get(k).to(self.device))
            out = self.agent.update(batch)
            logs.update(out)
        return logs

    def train(self):
        total_frames = int(self.cfg.train.total_frames)
        collect_per_iter = int(self.cfg.train.collect_steps_per_iter)
        updates_per_collect = int(self.cfg.train.updates_per_collect)

        while self.global_step < total_frames:
            # Anneal exploration parameters
            try:
                # Entropy coefficient: linear anneal from start -> entropy_anneal_to
                ent_start = float(getattr(self.cfg.train, "entropy_coef", 0.05))
                ent_to = float(getattr(self.cfg.train, "entropy_anneal_to", ent_start))
                ent_frames = max(1, int(getattr(self.cfg.train, "entropy_anneal_frames", total_frames)))
                frac = min(1.0, self.global_step / ent_frames)
                current_entropy_coef = ent_start + (ent_to - ent_start) * frac
                if hasattr(self.agent, "entropy_coef"):
                    self.agent.entropy_coef = float(current_entropy_coef)
                # Epsilon-greedy: linear decay from start -> epsilon_greedy_decay_to
                eps_start = float(getattr(self.cfg.train, "epsilon_greedy", 0.2))
                eps_to = float(getattr(self.cfg.train, "epsilon_greedy_decay_to", eps_start))
                eps_frames = max(1, int(getattr(self.cfg.train, "epsilon_greedy_decay_frames", total_frames)))
                frac_e = min(1.0, self.global_step / eps_frames)
                current_eps = eps_start + (eps_to - eps_start) * frac_e
                if hasattr(self.agent, "epsilon_greedy"):
                    self.agent.epsilon_greedy = float(current_eps)
                # Log occasionally
                if self.global_step % self._log_interval == 0:
                    try:
                        self.logger.add_scalar("exploration/entropy_coef", float(current_entropy_coef), self.global_step)
                        self.logger.add_scalar("exploration/epsilon_greedy", float(current_eps), self.global_step)
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
                steps = max(1, int(self.global_step))
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

            self._collect(collect_per_iter)
            logs = self._update(updates_per_collect)
            # logging (whitelisted)
            if logs and (self.global_step % self._log_interval == 0):
                # Map loss/entropy -> policy/entropy already done inside agent
                for k, v in logs.items():
                        self.logger.add_scalar(k, float(v), self.global_step)
                # 補助: 平均エピソードリターン（短窓）
                try:
                    import torch as _torch
                    if hasattr(self, "_ep_ret_env") and self._ep_ret_env is not None and hasattr(self, "_ep_len_env"):
                        lens = self._ep_len_env.clamp_min(1).to(dtype=_torch.float32)
                        mean_ret = float((self._ep_ret_env / lens).mean().item())
                        self.logger.add_scalar("env/mean_episode_return", mean_ret, self.global_step)
                except Exception:
                    pass
            # LLM系ログはダッシュボード対象外なので抑制
            # env/reward の代わりにエピソードリターンを用いる（別途終端時に出力）
            # checkpoint（任意、エージェント側の save 実装に委ねる）
            try:
                if self._save_every and (self.global_step % self._save_every == 0):
                    if hasattr(self.agent, "save"):
                        import os
                        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
                        path = f"{self.cfg.ckpt_dir}/ckpt_step_{self.global_step}.pt"
                        self.agent.save(path)
            except Exception:
                pass
            self.logger.flush()

