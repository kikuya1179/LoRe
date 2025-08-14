# LoRe: LLMのPolicy Priorを不確実性ゲートで安全に混合するDreamerV3拡張

不確実性に応じてLLMの方策事前分布を動的に注入し、スパース報酬環境での探索効率を向上させるモデルベース強化学習システム。

## 📋 概要

LoRe（Low-Regret LLM Prior）は、DreamerV3に対してLLM（大規模言語モデル）の方策事前分布を**必要なときのみ**注入する軽量拡張です。MiniGrid-DoorKey-5×5などのスパース報酬環境において、初期探索の効率化と安定した学習を両立します。

### 主要特徴

- **状態依存の不確実性ゲート**: β(s)による適応的なprior混合制御
- **KL逸脱制御**: ラグランジュ法による安全なprior注入
- **低侵襲実装**: DreamerV3の学習則に最小限の変更
- **実運用配慮**: API予算・クールダウン・キャッシュによるコスト抑制

## 🏗️ アーキテクチャ

```
LoRe/
├── main.py                 # メイン実行・統合制御
├── conf.py                 # 統合設定管理
├── agents/
│   └── dreamer_v3.py      # DreamerV3 + UncertaintyGate拡張
├── llm/
│   ├── controller.py      # LLM呼び出し制御・予算管理
│   ├── enhanced_adapter.py # MiniGrid特化アダプタ
│   ├── dsl_executor.py    # DSL実行エンジン
│   └── priornet.py        # 知識蒸留ネットワーク
├── utils/
│   ├── replay_buffer.py   # 成功バイアス付きリプレイバッファ
│   ├── metrics_aggregator.py # 包括的メトリクス集計
│   ├── health_monitor.py  # 健全性監視・異常検知
│   └── llm_adapter.py     # LLM統合基盤
├── experiments/
│   └── ablation.py        # A-F構成アブレーション実験
└── paper/
    ├── lore_paper.pdf     # 研究論文
    ├── result.txt         # 実験結果詳細
    └── sr_compare.png     # 成功率比較グラフ
```

## 🚀 クイックスタート

### 必要環境

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (推奨)
```

### インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# MiniGrid環境の確認
python -c "import minigrid; print('MiniGrid installed successfully')"
```

### 基本実行

```bash
# DreamerV3ベースライン (LLMなし)
python -m LoRe.main --env_id MiniGrid-DoorKey-5x5-v0 --total_steps 50000 --device cuda

# LoRe統合版 (LLM有効化には事前にconf.pyでLLMConfig.enabled=Trueに設定)
python -m LoRe.main --env_id MiniGrid-DoorKey-5x5-v0 --total_steps 50000 --device cuda --seed 42
```

## 🔬 理論的基盤

### 混合方策の定式化

LoReの核心は、DreamerV3の世界モデル由来方策とLLMのpriorを加法的に混合することです：

```
logits_mix(s) = logits_wm(s) + β(s) · stopgrad(logits_llm(s))
π_mix = softmax(logits_mix)
```

これは乗法的表現では以下のようになります：

```
π_mix(a|s) ∝ π_wm(a|s) · [π_llm(a|s)]^β(s)
```

### 不確実性ゲーティング β(s)

β値は状態の不確実性に応じて動的に調整されます：

```python
# 不確実性指標の組み合わせ
u(s) = w_H · H[π_wm(·|s)]     # 方策エントロピー
     + w_V · Var[V(s)]        # 価値分散
     + w_D · Disagreement(s)  # モデル分岐不一致

# β制御（シグモイド＋ヒステリシス）
β(s) = β_max · σ(κ(u(s) - τ))
```

### KL逸脱制御

混合方策が基底方策から過度に逸脱しないよう、KL制約を導入：

```python
# ラグランジュ法による制約
L_actor += λ · ReLU(KL(π_mix||π_wm) - δ_target)
λ ← clip([λ + η_λ(KL - δ_target)], 0, λ_max)
```

## 📊 実験結果

### MiniGrid-DoorKey-5×5での性能向上

**10k step近傍での比較**（Seed=42）：
- **RLのみ**: 成功率 ≈ 0.025
- **LoRe**: 成功率 ≈ 0.119（**+0.094の改善**）

**6k step時点**：
- **RLのみ**: 成功率 ≈ 0.045
- **LoRe**: 成功率 ≈ 0.20（**+0.16の改善**）

### 主要な観察事項

1. **早期立ち上がり**: LoReは中盤以降で恒常的な優位を維持
2. **適応的制御**: β値とKL制約の相互作用により、LLM依存度が自動調整
3. **コスト効率**: 12.7k stepで累積98回のLLM API呼び出し（必要時のみ発火）

## ⚙️ 設定

主要な設定は`conf.py`で管理されています：

### LLM設定

```python
@dataclass
class LLMConfig:
    enabled: bool = False             # LLM統合の有効化
    budget_total: int = 200          # API呼び出し予算
    cooldown_steps: int = 200        # 基本クールダウン
    success_cooldown_steps: int = 500 # 成功後クールダウン
    novelty_threshold: float = 0.1   # 新規性検知閾値
    td_error_threshold: float = 0.2  # TD誤差検知閾値
```

### LoRe制御設定

```python
@dataclass
class LoReConfig:
    beta_max: float = 0.3            # β最大値
    beta_warmup_steps: int = 5000    # βウォームアップ期間
    hysteresis_tau_low: float = 0.4  # ヒステリシス低閾値
    hysteresis_tau_high: float = 0.6 # ヒステリシス高閾値
    beta_dropout_p: float = 0.05     # βドロップアウト確率
    delta_target: float = 0.1        # KL制約目標値
    kl_lr: float = 1e-3             # KL制約学習率
    mix_in_imagination: bool = False # 潜在想像での混合
```

### 学習設定

```python
@dataclass
class TrainConfig:
    learning_rate: float = 1e-4      # 基本学習率
    entropy_coef: float = 0.01       # エントロピー係数
    epsilon_start: float = 0.3       # ε-greedy開始値
    epsilon_end: float = 0.1         # ε-greedy終了値
    tau_start: float = 2.0           # 温度アニーリング開始値
    tau_end: float = 1.0             # 温度アニーリング終了値
    replay_capacity: int = 100000    # リプレイバッファサイズ
    batch_size: int = 16             # バッチサイズ
    seq_len: int = 64                # シーケンス長
    warmup_steps: int = 5000         # ウォームアップステップ
```

## 📈 メトリクスと監視

### 基本学習メトリクス

- **方策**: entropy, prob_max, action分布
- **価値関数**: explained_variance, temporal_difference
- **世界モデル**: reconstruction_mse, reward_mae, PSNR

### LoRe専用メトリクス

- **β制御**: uncertainty_gate/avg_beta, beta_std
- **KL制御**: uncertainty_gate/avg_kl, lambda_kl
- **LLM使用**: llm_calls_used, cache_hit_rate
- **API効率**: budget_remaining, cooldown_status

### MiniGrid技能メトリクス

- **基本行動**: pickup_key_rate, door_toggle_rate, unlock_open_rate
- **効率性**: has_key_ratio, invalid_action_ratio
- **距離**: agent→key, key→door, door→goal (BFS距離の中央値)

## 🧪 実験

### 実験実行

```bash
# 完全なアブレーション実験
python -m LoRe.experiments.ablation --configs A B C D E --parallel 4

# 短縮実験（デバッグ用）
python -m LoRe.experiments.ablation --short --configs A D

# カスタム実験
python -m LoRe.main --env_id MiniGrid-DoorKey-5x5-v0 \
    --total_steps 100000 --seed 42 --device cuda
```

## 🔧 開発者向け情報

### 実装のポイント

1. **非侵襲性**: DreamerV3のコア学習ループは変更せず、行動選択時のみprior混合
2. **勾配制御**: `stopgrad(logits_llm)`により、LLM側に勾配を流さない
3. **温度整合**: LLMロジットの温度推定により、スケールを基底方策に合わせる
4. **フォールバック**: LLM呼び出し失敗時は基底方策にフォールバック

### 主要クラス

- **DreamerV3Agent**: 世界モデル + Actor-Critic + UncertaintyGate
- **LLMController**: 予算・クールダウン・キャッシュ管理
- **UncertaintyGate**: β(s)計算とヒステリシス制御
- **ReplayBuffer**: 成功バイアス付きサンプリング
- **MetricsAggregator**: 包括的メトリクス収集・出力

### カスタマイズポイント

```python
# β制御の重み調整
agent.uncertainty_gate.weights = {
    'entropy': 0.5,    # 方策エントロピーの重み
    'value_var': 0.3,  # 価値分散の重み
    'disagreement': 0.2 # モデル分岐不一致の重み
}

# KL制約の動的調整
agent.lore_cfg.delta_target = 0.15  # より厳しい制約
agent.lore_cfg.kl_lr = 2e-3         # より速い適応
```

## 🚨 トラブルシューティング

### よくある問題

**1. 方策エントロピーの低下**
```bash
[WARNING] Low entropy detected: 0.25 < 0.5
```
→ `entropy_coef`を増加、`tau_end`を調整、εアニーリングを緩和

**2. LLM API呼び出し失敗**
```bash
[ERROR] LLM request failed: timeout/rate_limit
```
→ `timeout_s`延長、`api_retries`増加、予算・クールダウン調整

**3. β値の異常**
```bash
[WARNING] Beta saturation: 0.95 > 0.8
```
→ `beta_max`削減、不確実性閾値調整、ウォームアップ期間延長

### デバッグモード

```bash
# 詳細ログ出力
export LORE_DEBUG=1
python -m LoRe.main --total_steps 10000

# ヘルス監視有効化
# conf.pyでlog.enable_health_monitor=True, health_verbose=True
```

## 📝 論文・引用

本実装は以下の論文に基づいています：

```bibtex
@article{lore2024,
  title={LoRe:不確実性ゲートと KL 制約による LLM 方策事前分布の安全混合},
  year={2025}
}
```

### 関連研究

- [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- [MiniGrid: Minimalistic Gridworld Environment](https://minigrid.farama.org/)
- [Plan2Explore: Self-Supervised Exploration via World Models](https://arxiv.org/abs/2005.05960)

## 🤝 貢献・ライセンス

### 貢献方法

1. 機能拡張・改善のPull Request歓迎
2. バグ報告・機能要求はIssue登録
3. 実験結果・知見の共有推奨

### ライセンス

各依存ライブラリのライセンスに従います。研究・教育目的での利用を推奨します。

---
