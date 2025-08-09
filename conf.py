from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    name: str = "CrafterReward-v1"  # Gymnasium registry name
    frame_skip: int = 4
    grayscale: bool = True
    image_size: int = 64
    frame_stack: int = 4
    norm_obs: bool = False
    action_repeat: int = 1


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 42
    total_frames: int = 1_000_00  # 100k default for quick tests
    init_random_frames: int = 10_000
    batch_size: int = 128
    updates_per_collect: int = 50
    collect_steps_per_iter: int = 200
    replay_capacity: int = 10_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_kl: float = 0.01  # for LLM prior regularization
    save_every_frames: int = 100_000
    log_interval: int = 1000
    max_episode_steps: Optional[int] = 2000
    # LLM
    use_llm: bool = False
    llm_model: str = "gemini-1.5-pro"


@dataclass
class ModelConfig:
    algo: str = "dreamer_v3"  # "dreamer" or "dreamer_v3"
    obs_channels: int = 1
    latent_dim: int = 256
    rssm_hidden: int = 200
    actor_hidden: int = 400
    critic_hidden: int = 400
    discrete_actions: bool = True
    # Optional LLM features fusion size
    llm_features_dim: int = 0


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log_dir: str = "runs/dreamer_crafter"
    ckpt_dir: str = "checkpoints"


def load_config() -> Config:
    """Load default config (extend with argparse/YAML if needed)."""
    return Config()

