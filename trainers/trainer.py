from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from utils.replay import make_replay


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

        # Reset environment
        td = self.env.reset()
        self._last_td = td

        self.global_step = 0

    def _get_obs_tensor(self, td) -> torch.Tensor:
        obs = td.get(self.obs_key)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        return obs.to(self.device, non_blocking=True)

    def _append_to_replay(self, prev_td, action, next_td):
        # Build a minimal tensordict sample with keys: observation, action, reward, done, next_observation
        from torchrl.data import TensorDict

        sample = TensorDict(
            {
                "observation": prev_td.get(self.obs_key).cpu(),
                "action": action.cpu(),
                "reward": next_td.get("reward").cpu(),
                "done": next_td.get("done").cpu(),
                "next": {
                    "observation": next_td.get(self.obs_key).cpu(),
                },
            },
            batch_size=[],
        )
        self.replay.add(sample)

    def _collect(self, steps: int):
        td = self._last_td
        for _ in range(steps):
            obs = self._get_obs_tensor(td)
            with torch.no_grad():
                action = self.agent.act(obs)
            # TorchRL envs expect action within a tensordict
            td.set("action", action.cpu())
            td = self.env.step(td)

            self._append_to_replay(self._last_td, action, td)
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
            self._collect(collect_per_iter)
            logs = self._update(updates_per_collect)
            # logging
            if logs:
                for k, v in logs.items():
                    self.logger.add_scalar(k, float(v), self.global_step)
            # simple reward log
            try:
                rew = float(self._last_td.get("reward").mean().item())
                self.logger.add_scalar("env/reward", rew, self.global_step)
            except Exception:
                pass
            self.logger.flush()

