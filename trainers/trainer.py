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

        # LLM adapter (Gemini 2.5 via google-generativeai)
        self.llm = LLMAdapter(
            LLMAdapterConfig(
                enabled=getattr(self.cfg.train, "use_llm", False),
                model=getattr(self.cfg.train, "llm_model", "gemini-2.5-flash-lite"),
                features_dim=getattr(self.cfg.model, "llm_features_dim", 0),
            )
        )

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
        # Build a minimal tensordict sample with keys: observation, action, reward, done, next_observation
        try:
            from tensordict import TensorDict  # TorchRL 0.9+ は独立パッケージ提供
        except Exception:  # pragma: no cover
            from torchrl.data import TensorDict  # 古いTorchRL向けフォールバック

        # 観測の抽出
        prev_obs = prev_td.get(self.obs_key)
        if prev_obs is None and ("next", self.obs_key) in prev_td.keys(True, True):
            prev_obs = prev_td.get(("next", self.obs_key))

        next_obs = next_td.get(("next", self.obs_key))
        if next_obs is None:
            next_obs = next_td.get(self.obs_key)

        # 報酬・終端は通常 'next' に格納
        rew = next_td.get(("next", "reward")) or next_td.get("reward")
        dn = next_td.get(("next", "done")) or next_td.get("done")
        if dn is None:
            term = next_td.get(("next", "terminated")) or next_td.get("terminated")
            trunc = next_td.get(("next", "truncated")) or next_td.get("truncated")
            if term is not None or trunc is not None:
                import torch as _torch
                t = (term if term is not None else 0)
                u = (trunc if trunc is not None else 0)
                dn = (t.to(dtype=_torch.bool) | u.to(dtype=_torch.bool)).to(dtype=_torch.float32)
            else:
                dn = None

        sample = TensorDict(
            {
                "observation": prev_obs.cpu(),
                "action": action.cpu(),
                "reward": (rew if rew is not None else 0.0).cpu() if hasattr(rew, "cpu") else torch.tensor(rew or 0.0),
                # done: terminated OR truncated（Gymnasium準拠）
                "done": (dn if dn is not None else torch.tensor(0.0)).cpu(),
                "next": {
                    "observation": next_obs.cpu(),
                },
            },
            batch_size=[],
        )
        # Optional LLM annotations
        if llm_out is not None:
            import torch as _torch
            if "prior_logits" in llm_out:
                sample.set("llm_prior_logits", _torch.tensor(llm_out["prior_logits"]).cpu())
            if "confidence" in llm_out:
                sample.set("llm_confidence", _torch.tensor(llm_out["confidence"]).cpu())
            if "mask" in llm_out:
                sample.set("llm_mask", _torch.tensor(llm_out["mask"]).cpu())
            if "features" in llm_out and getattr(self.cfg.model, "llm_features_dim", 0) > 0:
                sample.set("llm_features", _torch.tensor(llm_out["features"]).cpu())
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
            # LLM 事前/特徴の取得（任意、失敗は無視）
            llm_out = None
            try:
                if getattr(self.cfg.train, "use_llm", False):
                    obs_np = obs.detach().cpu().numpy()
                    llm_out = self.llm.infer(obs_np, num_actions=getattr(self.agent, "n_actions", 0))
            except Exception:
                llm_out = None

            # 初期ランダム行動期間
            with torch.no_grad():
                if self.global_step < self._random_warmup_steps:
                    import torch as _torch
                    n = int(getattr(self.agent, "n_actions", 1))
                    action = _torch.randint(low=0, high=n, size=(1, 1), device=self.device)
                else:
                    action = self.agent.act(obs)
            # TorchRL envs expect action within a tensordict
            td.set("action", action.cpu())
            td = self.env.step(td)

            # 集計
            try:
                r = float(td.get("reward").mean().item()) if "reward" in td.keys(True, True) else 0.0
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
            self._ep_return += r
            self._ep_len += 1
            # 行動ヒストグラム
            try:
                if self._action_hist is not None:
                    a = int(action.squeeze().item())
                    if 0 <= a < self._action_hist.numel():
                        self._action_hist[a] += 1
            except Exception:
                pass

            # エピソード終端時は自動リセット
            try:
                is_done = bool(td.get("done").item()) if "done" in td.keys(True, True) else False
            except Exception:
                is_done = False
            if is_done:
                # ロギング（エピソード単位）
                try:
                    self.logger.add_scalar("env/episode_return", self._ep_return, self.global_step)
                    self.logger.add_scalar("env/episode_length", float(self._ep_len), self.global_step)
                    # 簡易成功率: 報酬>0 を成功とみなす（Crafter本来の達成は環境Infoから取るのが理想）
                    self.logger.add_scalar("env/episode_success", 1.0 if self._ep_return > 0.0 else 0.0, self.global_step)
                    # 行動分布（直近エピソード）
                    if self._action_hist is not None:
                        total = int(self._action_hist.sum().item()) or 1
                        dist = {f"a{idx}": (int(v.item()) / total) for idx, v in enumerate(self._action_hist)}
                        self.logger.add_scalars("policy/action_dist", dist, self.global_step)
                except Exception:
                    pass
                self._ep_return = 0.0
                self._ep_len = 0
                if self._action_hist is not None:
                    self._action_hist.zero_()
                td = self.env.reset()

            self._append_to_replay(self._last_td, action, td, llm_out)
            self._last_td = td

            self.global_step += 1

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
            self._collect(collect_per_iter)
            logs = self._update(updates_per_collect)
            # logging
            if logs:
                for k, v in logs.items():
                    if self.global_step % self._log_interval == 0:
                        self.logger.add_scalar(k, float(v), self.global_step)
            # simple reward log
            try:
                if self.global_step % self._log_interval == 0:
                    rew = float(self._last_td.get("reward").mean().item())
                    self.logger.add_scalar("env/reward", rew, self.global_step)
            except Exception:
                pass
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

