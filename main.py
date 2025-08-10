import argparse
import os
from typing import Optional

import torch

from .conf import load_config, Config
from .utils.seed import set_seed
from .utils.logger import Logger
from .envs.crafter_env import make_crafter_env
from .agents.dreamer_v3 import DreamerV3Agent
from .trainers.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchRL Dreamer + Crafter Training")
    parser.add_argument("--total_frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def override_cfg(cfg: Config, args: argparse.Namespace) -> Config:
    if args.total_frames is not None:
        cfg.train.total_frames = args.total_frames
    if args.device is not None:
        cfg.train.device = args.device
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.seed is not None:
        cfg.train.seed = args.seed
    return cfg


def main() -> Optional[int]:
    args = parse_args()
    cfg = override_cfg(load_config(), args)

    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger = Logger(log_dir=cfg.log_dir)

    try:
        env = make_crafter_env(cfg.env)
    except Exception as e:
        import traceback
        print("[FATAL] Failed to create environment:", e)
        traceback.print_exc()
        return 1

    # obs_channels を env の前処理から推測（GrayScale + frame_stack 想定）。
    # 実観測から安全に推定する。
    try:
        td0 = env.reset()
        obs0 = td0.get("observation")
        if obs0 is None:
            obs0 = td0.get(("next", "observation"))
        if obs0 is not None:
            if obs0.dim() == 3:  # [C,H,W] or [H,W,C]
                ch = obs0.shape[0] if obs0.shape[0] in (1, 3, 4) else obs0.shape[-1]
            elif obs0.dim() == 4:  # [B,C,H,W] など
                ch = obs0.shape[1]
            else:
                ch = getattr(cfg.model, "obs_channels", 1)
            cfg.model.obs_channels = int(ch)
    except Exception:
        pass

    agent = DreamerV3Agent(
        cfg.model,
        action_spec=env.action_spec,
        device=device,
        lr=cfg.train.learning_rate,
        gamma=cfg.train.gamma,
        entropy_coef=getattr(cfg.train, "entropy_coef", 0.05),
        epsilon_greedy=getattr(cfg.train, "epsilon_greedy", 0.0),
    )
    # expose lambda_kl to the agent for LLM prior regularization
    try:
        agent.lambda_kl = float(cfg.train.lambda_kl)
    except Exception:
        agent.lambda_kl = 0.0

    trainer = Trainer(
        env=env,
        agent=agent,
        logger=logger,
        cfg=cfg,
        device=device,
    )

    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

