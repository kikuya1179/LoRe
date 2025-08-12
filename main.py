import argparse
import os
import warnings
from typing import Optional

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchrl")
warnings.filterwarnings("ignore", message=".*gym.*")
warnings.filterwarnings("ignore", message=".*Gym.*")

import torch

from .conf import load_config, Config
from .utils.seed import set_seed
from .utils.logger import Logger
from .envs.crafter_env import make_crafter_env
from .agents.dreamer_v3 import DreamerV3Agent
from .trainers.trainer import Trainer
try:
    from .trainers.async_trainer import AsyncTrainer  # type: ignore
except Exception:
    AsyncTrainer = None  # type: ignore
from .cuda_optimizer import setup_cuda_optimization, optimize_batch_size, warmup_cuda, print_gpu_status
try:
    from .trainers.enhanced_trainer import EnhancedTrainer  # type: ignore
except Exception:  # pragma: no cover
    EnhancedTrainer = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchRL Dreamer + Crafter Training")
    parser.add_argument("--total_frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # Performance-related overrides
    parser.add_argument("--updates_per_collect", type=int, default=None)
    parser.add_argument("--collect_steps_per_iter", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--frame_stack", type=int, default=None)
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
    # Performance-related overrides
    if getattr(args, "updates_per_collect", None) is not None:
        cfg.train.updates_per_collect = int(args.updates_per_collect)
    if getattr(args, "collect_steps_per_iter", None) is not None:
        cfg.train.collect_steps_per_iter = int(args.collect_steps_per_iter)
    if getattr(args, "batch_size", None) is not None:
        cfg.train.batch_size = int(args.batch_size)
    if getattr(args, "num_envs", None) is not None:
        cfg.env.num_envs = int(args.num_envs)
    if getattr(args, "image_size", None) is not None:
        cfg.env.image_size = int(args.image_size)
    if getattr(args, "frame_stack", None) is not None:
        cfg.env.frame_stack = int(args.frame_stack)
    return cfg


def main() -> Optional[int]:
    args = parse_args()
    cfg = override_cfg(load_config(), args)

    # Apply runtime threading limits from config (replaces shell env exports)
    try:
        import os as _os
        # External libs (MKL/NumExpr/OpenMP)
        _os.environ["OMP_NUM_THREADS"] = str(int(getattr(cfg.train, "omp_num_threads", 1)))
        _os.environ["MKL_NUM_THREADS"] = str(int(getattr(cfg.train, "mkl_num_threads", 1)))
        _os.environ["NUMEXPR_MAX_THREADS"] = str(int(getattr(cfg.train, "numexpr_max_threads", 1)))
    except Exception:
        pass

    set_seed(cfg.train.seed)

    # Setup CUDA with comprehensive optimization
    # process identity for prints
    try:
        import os as _os, platform as _pf, time as _time
        _proc = f"pid={_os.getpid()}@{_pf.node()}"
    except Exception:
        _proc = "pid=?"
    if cfg.train.device == "cuda":
        if not torch.cuda.is_available():
            print("[FATAL] CUDA requested but not available!")
            return 1
        
        # Apply CUDA optimizations
        cuda_status = setup_cuda_optimization()
        print_gpu_status()
        print(f"[proc:{_proc}] [GPU] Applied optimizations: {', '.join(cuda_status['optimizations'])}")
        
        device = torch.device("cuda")
        
        # Optimize batch size based on GPU memory
        cfg.train.batch_size = optimize_batch_size(cfg.train.batch_size, device)
        
        # CUDA warmup for faster first iteration
        warmup_cuda(device)
        
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU device")
    # Torch thread control (in both CPU and CUDA cases)
    try:
        torch.set_num_threads(int(getattr(cfg.train, "torch_num_threads", 1)))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(int(getattr(cfg.train, "torch_num_interop_threads", 1)))
    except Exception:
        pass
    # Allow only requested metrics
    allowed = {
        # env
        "env/episode_return",
        "env/mean_episode_return",
        "env/success_rate",
        "env/score_percent",
        "env/crafter_score",
        # losses & policy
        "loss/model_recon",
        "loss/model_reward",
        "loss/value",
        "loss/kl_divergence",
        "policy/entropy",
        # learning dynamics / optimizer
        "advantage/mean",
        "advantage/std",
        "value/mean",
        "value/std",
        "reward/batch_mean",
        "reward/batch_std",
        "value/td_abs_mean",
        "value/td_std",
        "value/explained_variance",
        "world/psnr_db",
        "optim/grad_global_norm",
        "optim/lr",
    }
    logger = Logger(log_dir=cfg.log_dir, allowed_tags=allowed)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

    # Create agent
    agent = DreamerV3Agent(
        cfg.model,
        action_spec=env.action_spec,
        device=device,
        lr=cfg.train.learning_rate,
        gamma=cfg.train.gamma,
        entropy_coef=getattr(cfg.train, "entropy_coef", 0.05),
        # ε-greedy は AsyncTrainer 側の温度ソフトマックスと重複させない
        epsilon_greedy=0.0,
    )
    
    # GPU memory check
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU] Model loaded - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
    # Set lambda_kl for LLM prior regularization
    try:
        agent.lambda_kl = float(cfg.train.lambda_kl)
    except Exception:
        agent.lambda_kl = 0.0

    # Choose trainer
    use_enhanced = bool(getattr(cfg.train, "use_enhanced_trainer", False)) and (EnhancedTrainer is not None)
    # Prefer AsyncTrainer when available and CUDA device is used
    prefer_async = True
    if prefer_async and AsyncTrainer is not None:
        trainer_cls = AsyncTrainer  # type: ignore[misc]
    else:
        trainer_cls = EnhancedTrainer if use_enhanced else Trainer
    trainer = trainer_cls(  # type: ignore[misc]
        env=env,
        agent=agent,
        logger=logger,
        cfg=cfg,
        device=device,
    )

    # Final GPU verification before training
    if device.type == "cuda":
        print(f"\n[proc:{_proc}] [GPU] Starting training on {torch.cuda.get_device_name(0)}")
        print(f"[proc:{_proc}] [GPU] Batch size: {cfg.train.batch_size}")
        print(f"[proc:{_proc}] [GPU] Memory available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2:.1f}MB")
        # Reset wall clock for clean step=1000 cadence from 0
        try:
            logger._t0 = _time.time()
        except Exception:
            pass
        
        # Test GPU operations
        with torch.no_grad():
            test_tensor = torch.randn(100, 100, device=device)
            _ = torch.mm(test_tensor, test_tensor)
        print(f"[proc:{_proc}] [GPU] GPU operations verified")
    
    trainer.train()
    # Final hardware report (GPU/CPU/Memory/OS)
    try:
        import platform, psutil
        print("\n===== HARDWARE SUMMARY =====")
        print("OS:", platform.platform())
        print("Python:", platform.python_version())
        print("CPU:", platform.processor())
        try:
            cpu_count = psutil.cpu_count(logical=True)
            load = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory()
            print(f"CPU Cores(logical): {cpu_count}, Load: {load}%")
            print(f"RAM: {mem.used/1024**3:.2f}GB used / {mem.total/1024**3:.2f}GB")
        except Exception:
            pass
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            cached = torch.cuda.memory_reserved() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Allocated: {allocated:.1f}MB, Reserved: {cached:.1f}MB, Peak: {max_allocated:.1f}MB")
            torch.cuda.empty_cache()
        print("============================\n")
    except Exception:
        pass
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

