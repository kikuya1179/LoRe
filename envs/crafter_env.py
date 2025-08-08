from dataclasses import asdict
from typing import Optional


def make_crafter_env(env_cfg) -> "torchrl.envs.EnvBase":  # type: ignore[name-defined]
    """Create a Crafter environment wrapped as a TorchRL TransformedEnv.

    This attempts Gymnasium first (CrafterReward-v1 registered by "crafter" package),
    then falls back to direct crafter env construction if needed.
    Applies standard Dreamer-style transforms (resize, grayscale, frame-stack).
    """
    try:
        import gymnasium as gym
    except Exception as e:  # pragma: no cover - informative error
        raise RuntimeError("Gymnasium is required to build the Crafter env") from e

    try:
        from torchrl.envs import GymEnv
        from torchrl.envs.transforms import (
            TransformedEnv,
            Compose,
            GrayScale,
            ToTensorImage,
            Resize,
            CatFrames,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError("TorchRL is required for env wrappers and transforms") from e

    # Build base gym env
    env_id = getattr(env_cfg, "name", "CrafterReward-v1")
    kwargs = {}
    if getattr(env_cfg, "max_episode_steps", None) is not None:
        kwargs["max_episode_steps"] = env_cfg.max_episode_steps

    try:
        base = gym.make(env_id)
    except Exception as e:
        # Helpful message if crafter not installed
        raise RuntimeError(
            f"Failed to make gym env '{env_id}'. Ensure 'crafter' is installed and registered."
        ) from e

    env = GymEnv(base)

    tfs = []
    # Convert to tensor image first
    tfs.append(ToTensorImage())
    if getattr(env_cfg, "grayscale", True):
        tfs.append(GrayScale())
    if getattr(env_cfg, "image_size", 64) is not None:
        tfs.append(Resize(env_cfg.image_size, env_cfg.image_size))
    # Frame stacking (post grayscale/resize)
    fs = getattr(env_cfg, "frame_stack", 4)
    if fs and fs > 1:
        tfs.append(CatFrames(N=fs, dim=-3))

    env = TransformedEnv(env, Compose(*tfs))
    return env

