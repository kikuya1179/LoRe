from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    name: str = "CrafterReward-v1"  # Gymnasium registry name
    frame_skip: int = 1
    grayscale: bool = True
    image_size: int = 48
    frame_stack: int = 3
    norm_obs: bool = False
    action_repeat: int = 1
    num_envs: int = 4


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 42
    total_frames: int = 1_000_000  # Fixed typo
    init_random_frames: int = 10_000
    batch_size: int = 256
    updates_per_collect: int = 150
    collect_steps_per_iter: int = 1000
    replay_capacity: int = 10_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_kl: float = 0.01  # for LLM prior regularization
    # Exploration settings
    # Exploration (強化): 初動を作るため増強
    entropy_coef: float = 0.08
    epsilon_greedy: float = 0.2
    # Annealing schedules (optional)
    entropy_anneal_to: float = 0.02
    entropy_anneal_frames: int = 600_000
    epsilon_greedy_decay_to: float = 0.05
    epsilon_greedy_decay_frames: int = 800_000
    # Intrinsic reward (RND)
    use_intrinsic: bool = False
    intrinsic_coef: float = 0.2
    intrinsic_norm: bool = True
    save_every_frames: int = 100_000
    log_interval: int = 4000
    max_episode_steps: Optional[int] = 2000
    intrinsic_update_every: int = 8
    
    # Enhanced LLM Configuration
    use_llm: bool = False
    use_enhanced_trainer: bool = False
    llm_model: str = "gemini-2.5-flash-lite"
    llm_use_cli: bool = False
    llm_batch_size: int = 8
    llm_cache_size: int = 1000
    llm_use_dsl: bool = True
    llm_timeout_s: float = 10.0
    llm_api_retries: int = 3
    # LLM call throttling
    llm_beta_call_threshold: float = 0.5
    llm_cooldown_steps: int = 1000
    llm_call_budget_total: int = 500
    llm_call_on_episode_boundary: bool = True
    llm_min_macro_interval: int = 1500
    llm_use_cache: bool = True
    # Long-run scheduling (10M+)
    llm_calls_per_million: int = 500
    llm_progress_hardening: bool = True
    # PriorNet distillation
    llm_priornet_enabled: bool = True
    llm_priornet_update_every: int = 50
    llm_priornet_hidden: int = 256
    llm_priornet_lr: float = 1e-3
    llm_priornet_temp: float = 2.0
    # LLM context payloads
    llm_send_latent: bool = True
    llm_latent_dim: int = 32
    llm_send_summary: bool = True
    llm_send_image: bool = True
    llm_image_size: int = 16
    llm_image_single_channel: bool = True
    
    # LLM Triggers
    llm_novelty_threshold: float = 0.1
    llm_td_error_threshold: float = 0.5
    llm_plateau_frames: int = 150
    
    # Reward Shaping
    reward_shaping_beta: float = 0.1
    reward_shaping_clip: float = 0.2
    
    # HER (Hindsight Experience Replay)
    use_her: bool = False
    her_ratio: float = 0.0
    
    # Priority Replay
    use_priority_replay: bool = False
    priority_alpha: float = 0.0
    priority_beta: float = 0.0
    
    # Distillation
    use_distillation: bool = False
    distill_initial_alpha: float = 0.0
    distill_final_alpha: float = 0.01
    distill_every: int = 1000
    
    # LoRe Synthetic Experience Parameters
    synthetic_ratio_max: float = 0.25  # Maximum ratio of synthetic data in replay buffer
    synthetic_weight_decay: float = 0.99  # Decay factor for synthetic data importance
    synthetic_rollout_length: int = 5  # Maximum length of synthetic rollouts
    synthetic_confidence_threshold: float = 0.3  # Minimum confidence to generate synthetic data
    synthetic_success_threshold: float = 0.1  # Reward threshold for successful synthetic plans
    synthetic_execution_prob: float = 0.0  # Disabled for speed
    synthetic_generation_interval: int = 100  # Steps between synthetic generation attempts


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
    
    # LoRe Uncertainty Gate Parameters
    beta_max: float = 0.3  # Maximum β value for LLM mixing
    delta_target: float = 0.1  # Target KL divergence threshold
    kl_lr: float = 1e-3  # Learning rate for Lagrange multiplier
    uncertainty_threshold: float = 0.4  # Threshold for uncertainty-based gating
    use_value_ensemble: bool = False  # Enable value ensemble for uncertainty estimation
    
    # LoRe Synthetic Data Parameters
    lambda_bc: float = 0.1  # Behavioral cloning regularization coefficient for synthetic data
    
    # LoRe Option System Parameters
    max_options: int = 8  # Maximum number of options/skills
    option_generation_interval: int = 500  # Steps between skill generation attempts
    skill_confidence_threshold: float = 0.4  # Minimum confidence for generated skills
    enable_hierarchical_options: bool = False  # Enable option system


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










