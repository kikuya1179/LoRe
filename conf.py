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
    total_frames: int = 1_000_000  # Fixed typo
    init_random_frames: int = 10_000
    batch_size: int = 256
    updates_per_collect: int = 100
    collect_steps_per_iter: int = 1000
    replay_capacity: int = 10_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_kl: float = 0.01  # for LLM prior regularization
    entropy_coef: float = 0.02  # encourage exploration
    epsilon_greedy: float = 0.1  # random action probability in act()
    # Intrinsic reward (RND)
    use_intrinsic: bool = False
    intrinsic_coef: float = 0.1
    intrinsic_norm: bool = True
    save_every_frames: int = 100_000
    log_interval: int = 1000
    max_episode_steps: Optional[int] = 2000
    
    # Enhanced LLM Configuration
    use_llm: bool = False
    llm_model: str = "gemini-2.5-flash-lite"
    llm_use_cli: bool = True
    llm_batch_size: int = 8
    llm_cache_size: int = 1000
    llm_use_dsl: bool = True
    llm_timeout_s: float = 2.5
    
    # LLM Triggers
    llm_novelty_threshold: float = 0.1
    llm_td_error_threshold: float = 0.5
    llm_plateau_frames: int = 10000
    
    # Reward Shaping
    reward_shaping_beta: float = 0.1
    reward_shaping_clip: float = 0.2
    
    # HER (Hindsight Experience Replay)
    use_her: bool = True
    her_ratio: float = 0.8
    
    # Priority Replay
    use_priority_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    
    # Distillation
    use_distillation: bool = True
    distill_initial_alpha: float = 0.3
    distill_final_alpha: float = 0.01
    distill_every: int = 1000


@dataclass
class ModelConfig:
    algo: str = "dreamer_v3"  # "dreamer" or "dreamer_v3"
    obs_channels: int = 1
    latent_dim: int = 256
    rssm_hidden: int = 200
    actor_hidden: int = 400
    critic_hidden: int = 400
    discrete_actions: bool = True
    
    # Enhanced LLM Integration
    llm_features_dim: int = 16  # Enable LLM feature fusion
    max_llm_features: int = 32
    feature_normalize_range: float = 3.0  # [-3, 3] normalization


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

