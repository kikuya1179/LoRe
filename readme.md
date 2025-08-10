# 🚀 LoRe: LLM-enhanced Reinforcement Learning

**LoRe**は大規模言語モデル（LLM）の知識と計画力を強化学習に統合する完全なフレームワークです。DreamerV3をベースに、LLMを「助言者」として3つの経路で活用し、サンプル効率と探索性能を大幅に向上させます。

![LoRe Architecture](https://img.shields.io/badge/LoRe-LLM%2BRL%20Integration-blue)
[![Tests](https://img.shields.io/badge/tests-passing-green)](./test_integration.py)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.6%2B-orange)](https://pytorch.org)

## ✨ 主な特徴

### 🧠 **3つのLLM統合経路（完全実装）**

1. **🔄 A) Replay拡張（合成経験）**
   - LLMが提案するプラン/マクロ行動を実行
   - 世界モデルで合成遷移を生成してリプレイバッファに追加
   - 行動クローニング正則化で分布シフトを抑制
   - 重み付きサンプリングで合成データ比率を制御（≤25%）

2. **🎯 B) Policy Prior（不確実性ベースバイアス）**
   - 世界モデル方策に不確実性で制御されたLLM助言を注入
   - `logits_mix = logits_wm + β(s) * stopgrad(logits_llm)`
   - β値は状態エントロピー・価値分散・モデル不一致で適応
   - 目標KL制約でLLMの過度な影響を防止

3. **🏗️ C) Option層（階層的スキル）**
   - LLMが「スキル名＋実装」を生成
   - 拡張行動空間：`𝒜' = 𝒜_primitive ∪ {option_m}`
   - Call-and-Return実行で複雑タスクを分解
   - 性能ベースの自動スキル管理（悪いスキルは削除）

### 🎮 **環境・モデル統合**
- **Crafter環境**: 2D Minecraft様の複雑ゲーム環境
- **DreamerV3**: 世界モデルベースの強化学習（RSSM + λ-returns）
- **TorchRL**: 高性能なRL実装基盤
- **Windows対応**: メモリ効率化とクロスプラットフォーム

### 🔧 **設定駆動 & 拡張性**
- 完全設定駆動（`conf.py`）
- モジュラー設計で個別有効化可能
- 包括的モニタリング & TensorBoard統合

---

## 📋 要件

```bash
Python 3.10+
PyTorch 2.6+
TorchRL 0.9+
Crafter
TensorBoard
```

**推奨環境**: GPU（CUDA対応）、8GB+ RAM

---

## 🚀 クイックスタート

### 1. インストール

```powershell
# 仮想環境作成
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 依存関係インストール
python -m pip install --upgrade pip
pip install -r LoRe/requirements.txt
```

### 2. 基本実行（DreamerV3のみ）

```powershell
# Windows推奨設定
$Env:LORE_REPLAY_BACKEND = "tensor"

# 基本学習実行
python -m LoRe.main --total_frames 100000 --device cuda --log_dir runs/dreamer_basic
```

### 3. LoRe統合実行（全機能有効）

```powershell
# LLM API設定（Gemini）
$Env:GEMINI_API_KEY = "your-api-key-here"

# LoRe統合学習
python -m LoRe.main `
  --total_frames 500000 `
  --device cuda `
  --log_dir runs/lore_full `
  --use_llm `
  --enable_synthetic_replay `
  --enable_hierarchical_options
```

### 4. TensorBoard監視

```powershell
tensorboard --logdir runs/lore_full
```

---

## ⚙️ 設定システム

### 基本設定（`LoRe/conf.py`）

```python
@dataclass
class TrainConfig:
    # 基本学習
    total_frames: int = 1_000_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    
    # LLM統合
    use_llm: bool = False
    llm_model: str = "gemini-2.5-flash-lite"
    
    # A) 合成リプレイ
    synthetic_ratio_max: float = 0.25
    synthetic_execution_prob: float = 0.2
    
    # B) 不確実性ゲート  
    beta_max: float = 0.3
    delta_target: float = 0.1
    
    # C) 階層的オプション
    max_options: int = 8
    option_generation_interval: int = 500
```

### 段階的有効化

```python
# 1. 基本DreamerV3
config = load_config()

# 2. + Policy Prior (B)
config.train.use_llm = True
config.model.beta_max = 0.3

# 3. + Synthetic Replay (A) 
config.train.synthetic_ratio_max = 0.25

# 4. + Hierarchical Options (C)
config.model.enable_hierarchical_options = True
config.model.max_options = 8
```

---

## 📊 モニタリング & 検証

### パフォーマンス指標

```python
# TensorBoard メトリクス
- env/episode_return          # エピソード報酬
- loss/policy, loss/value     # 方策・価値損失
- uncertainty_gate/avg_kl     # KL制約状況
- synthetic/ratio             # 合成データ比率
- options/avg_success_rate    # オプション成功率
- llm_adapter/cache_hit_rate  # LLM効率性
```

---

## 🏗️ アーキテクチャ詳細

### フレームワーク構成

```
LoRe/
├── agents/
│   ├── dreamer_v3.py              # ベースDreamerV3実装
│   └── dreamer_v3_options.py      # 階層的DreamerV3
├── options/                       # Option層システム
│   ├── option_framework.py        # オプション管理・実行
│   └── llm_skill_generator.py     # LLMスキル生成
├── utils/
│   ├── synthetic_replay.py        # 拡張リプレイバッファ
│   ├── synthetic_generator.py     # 合成経験生成
│   └── llm_adapter.py            # LLMインタフェース
├── trainers/
│   ├── trainer.py                 # 基本トレーナー
│   └── enhanced_trainer.py        # LoRe統合トレーナー
└── envs/
    └── crafter_env.py             # Crafter環境ラッパー
```

### LoRe統合の核心理論

**目標**: 世界モデル `p_θ` を土台に、LLMを助言者として3経路で統合

1. **Replay拡張**: `w = (1-is_synth) + is_synth * w_synth`
2. **Policy Prior**: `logits_mix = logits_wm + β(s) * stopgrad(logits_llm)`  
3. **Option化**: `𝒜' = 𝒜_primitive ∪ {option_m}` で階層実行

**安全性**: 分布シフト対策（KL制約・BC正則化・重要度制限）でLLMが暴走しない設計

---

## 📊 実験・結果

### ベンチマーク環境
- **Crafter**: 2D Minecraft様の複合タスク環境
- **評価指標**: 成功率、サンプル効率、エピソード長、学習安定性

### 期待される改善
- **サンプル効率**: 30-50%向上（合成経験による探索加速）
- **成功率**: 20-40%向上（LLM知識による方向性）
- **収束速度**: 2-3倍高速化（階層分解による効率性）

### アブレーション研究
1. ベースライン（DreamerV3）
2. +Policy Prior のみ
3. +Synthetic Replay のみ  
4. +Option層のみ
5. LoRe完全版（A+B+C）

---

## 🛠️ 高度な使用法

### カスタムLLMアダプター

```python
from LoRe.utils.llm_adapter import LLMAdapter

class CustomLLMAdapter(LLMAdapter):
    def infer(self, obs_np, num_actions):
        # カスタムLLM呼び出しロジック
        return {
            'prior_logits': your_logits,
            'confidence': [confidence],
            'features': your_features
        }
```

### 独自スキル定義

```python
from LoRe.options import OptionSpec

skill = OptionSpec(
    option_id="custom_gather",
    name="Advanced Wood Gathering",
    description="Optimized wood collection sequence",
    primitive_actions=[2, 5, 1, 5, 3, 5],  # right,do,left,do,up,do
    expected_duration=6,
    confidence=0.9,
)
```

### ファインチューニング

```python
# 不確実性ゲートの調整
config.model.beta_max = 0.5          # LLM影響を強化
config.model.delta_target = 0.05     # KL制約を厳格化
config.model.uncertainty_threshold = 0.3  # より積極的なゲート

# 合成データ制御  
config.train.synthetic_ratio_max = 0.3     # 合成比率上限
config.train.synthetic_weight_decay = 0.95 # 重み減衰を強化

# オプション管理
config.model.max_options = 12              # スキル容量拡大
config.train.option_generation_interval = 200  # 生成頻度向上
```

---

## 🔍 トラブルシューティング

### 一般的な問題

**メモリ不足**
```powershell
# リプレイ容量削減
$Env:LORE_REPLAY_BACKEND = "tensor"
# またはconf.pyで replay_capacity を削減
```

**学習不安定**
```python
# KL制約を厳格化
config.model.delta_target = 0.05
# 合成データ比率削減
config.train.synthetic_ratio_max = 0.15
```

**LLMレスポンス遅延**
```python  
config.train.llm_timeout_s = 1.5      # タイムアウト短縮
config.train.llm_cache_size = 2000     # キャッシュ拡大
```

**オプション性能低下**
```python
config.train.skill_confidence_threshold = 0.6  # 生成基準厳格化
# 不要スキルの手動削除も可能
```

### デバッグモード

```python
# 詳細ログ有効化
config.train.log_interval = 100

# テスト用短時間実行
python -m LoRe.main --total_frames 5000 --device cpu
```

---

## 🤝 コントリビュート

### 開発セットアップ

```bash
# 開発環境
pip install -e LoRe/
pip install pytest black flake8

# テスト実行
pytest test_*.py

# コード整形
black LoRe/
```

### 拡張ポイント
- **新環境**: `envs/`にアダプター追加
- **新LLM**: `utils/llm_adapter.py`を拡張
- **新スキル**: `options/llm_skill_generator.py`にテンプレート追加
- **新指標**: `trainers/enhanced_trainer.py`にメトリクス追加

---

## 📚 引用・参考文献

```bibtex
@article{lore2024,
  title={LoRe: LLM-enhanced Reinforcement Learning with Multi-Path Integration},
  author={LoRe Team},
  journal={arXiv preprint},
  year={2024}
}
```

**関連研究**:
- DreamerV3: Hafner et al. (2023)
- Crafter: Hafner (2021) 
- Hierarchical RL: Sutton et al. (1999)
- LLM-guided RL: Reed et al. (2022)

---

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照

---

## 🚀 今後の展開

### v1.1 予定機能
- [ ] **マルチ環境対応**: MiniGrid, ALE, MuJoCo
- [ ] **モデル並列化**: 大規模環境向け分散学習
- [ ] **対話的スキル編集**: 人間-LLM協調スキル改良
- [ ] **メタ学習統合**: 環境間知識転移

### v2.0 ビジョン
- [ ] **マルチモーダルLLM**: 視覚-言語統合
- [ ] **オンライン学習**: リアルタイム環境適応
- [ ] **説明可能性**: 意思決定の解釈機能
- [ ] **安全性保証**: フォーマル検証付きRL

---

**🎯 LoRe**: 大規模言語モデルの知恵を強化学習に注入し、人工知能の新たな可能性を切り開きます。

[![GitHub](https://img.shields.io/badge/GitHub-LoRe-blue?logo=github)](https://github.com/your-repo/LoRe)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](./docs/)
[![Discord](https://img.shields.io/badge/Discord-Community-7289da?logo=discord)](https://discord.gg/your-discord)