import argparse
import os
from typing import Optional

import torch

from conf import load_config, Config
from utils.seed import set_seed
from utils.logger import Logger
from envs.crafter_env import make_crafter_env
from agents.dreamer import DreamerAgent
from trainers.trainer import Trainer


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
        print("[FATAL] Failed to create environment:", e)
        return 1

    agent = DreamerAgent(cfg.model, action_spec=env.action_spec, device=device, lr=cfg.train.learning_rate)

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

