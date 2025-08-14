from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    enabled: bool = True
    budget_total: int = 500
    cooldown_steps: int = 200
    success_cooldown_steps: int = 500
    novelty_threshold: float = 0.1
    td_error_threshold: float = 0.1
    plateau_frames: int = 300
    temperature_estimation_samples: int = 100


@dataclass
class LoReConfig:
    beta_max: float = 0.3
    beta_warmup_steps: int = 5000
    hysteresis_tau_low: float = 0.4
    hysteresis_tau_high: float = 0.6
    beta_dropout_p: float = 0.05
    delta_target: float = 0.1
    kl_lr: float = 1e-3
    uncertainty_threshold: float = 0.4
    use_value_ensemble: bool = False
    use_priornet: bool = False
    mix_in_imagination: bool = False  # 本番はまずFalseで安定化
    # 想像空間でのLoRe混合の安全上限
    beta_imagine_max: float = 0.1
    # KL制約の段階適用（学習アップデート数で切替）
    kl_phase_switch_updates: int = 10000
    use_prox_kl: bool = True
    use_base_kl: bool = True
    # LLMロジット整流
    llm_center_logits: bool = True
    llm_temperature: float = 2.0


@dataclass
class LogConfig:
    metrics_interval: int = 250
    save_interval: int = 2000
    health_check_interval: int = 2000
    health_warmup_steps: int = 2000
    health_verbose: bool = False
    enable_health_monitor: bool = False


@dataclass
class EnvConfig:
    id: str = "MiniGrid-DoorKey-5x5-v0"
    image_size: int = 64
    grayscale: bool = True


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 42
    total_steps: int = 50_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    entropy_coef: float = 0.04
    # Exploration schedule
    epsilon_start: float = 0.3
    epsilon_end: float = 0.10
    epsilon_anneal_steps: int = 10_000
    # Softmax temperature schedule (for sampling)
    tau_start: float = 1.2
    tau_end: float = 1.2
    tau_anneal_steps: int = 5_000
    # Actor warmup schedule
    actor_warmup_steps: int = 8_000
    actor_anneal_steps: int = 8_000
    # Replay / sequence training
    replay_capacity: int = 100_000
    seq_len: int = 16
    warmup_steps: int = 1_000
    updates_per_step: int = 1
    # Success-biased replay (non-invasive boost)
    replay_success_boost: float = 5.0
    replay_back_steps: int = 10
    # Periodic evaluation (does not affect learning)
    eval_interval: int = 2000
    eval_episodes: int = 5
    eval_seed: int = 0
    # Prefill before training (alias of warmup for clarity in logs)
    min_prefill_steps: int = 3000
    # Reward shaping (event-based; safe defaults)
    shaping_enabled: bool = True
    shaping_key_bonus: float = 0.05
    shaping_door_bonus: float = 0.10
    shaping_invalid_penalty: float = 0.001
    shaping_stationary_penalty: float = 0.001
    shaping_stationary_N: int = 10
    shaping_use_potential: bool = False
    shaping_potential_cap: float = 0.01
    # valid限定ノイズ（εの一様混合は有効アクションに限定）
    valid_only_noise: bool = True


@dataclass
class ModelConfig:
    obs_channels: int = 1
    latent_dim: int = 256
    llm_features_dim: int = 0
    # Synthetic/BC (kept for compatibility in agent)
    lambda_bc: float = 0.1


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    lore: LoReConfig = field(default_factory=LoReConfig)
    log: LogConfig = field(default_factory=LogConfig)
    log_dir: str = "runs/minigrid_dreamer"
    ckpt_dir: str = "checkpoints"


def load_config() -> Config:
    return Config()

 
