# LoRe

DreamerV3 + TorchRL + Crafter に、任意で LLM 事前（Gemini）を組み合わせる最小実装です。

### 主な特徴
- 設定駆動の前処理（GrayScale/Resize/FrameStack）と観測キーの自動正規化
- 省メモリ＆Windows 対応のリプレイ（memmap 回避オプション）
- TD(0) 整合の強化、安定した学習ループ、任意のチェックポイント保存
- LLM 事前分布（KL 項）・特徴連結の配線下地（無効時コストゼロ）

---

## 要件
- Python 3.10 推奨
- PyTorch 2.6 系 + TorchRL 0.9 系（`LoRe/requirements.txt` 参照）
- GPU 推奨（`--device cuda`）。CPU でも動作可（遅くなります）

## インストール
仮想環境の作成（例: venv）と依存インストール:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r LoRe/requirements.txt
```
Windows 固有の注意は `LoRe/INSTALL_WINDOWS.txt` を参照してください。

## クイックスタート
最短で動かす（ログは TensorBoard に出力）
```powershell
$Env:LORE_REPLAY_BACKEND = "tensor"   # Windows 推奨（memmap回避）
python -m LoRe.main --total_frames 2000 --device cuda --log_dir runs/dreamer_crafter
```

TensorBoard:
```powershell
tensorboard --logdir runs/dreamer_crafter
```

## 使い方（CLI）
以下は代表的な上書き例です。
```powershell
# 総フレーム数・デバイス・ログ先を指定
python -m LoRe.main --total_frames 100000 --device cuda --log_dir runs/dreamer_crafter

# CPU 実行（遅くなります）
python -m LoRe.main --total_frames 10000 --device cpu
```

## 設定
既定値は `LoRe/conf.py` で定義されています。主な項目:
- EnvConfig
  - `name`: Crafter の環境名（例: `CrafterReward-v1`）
  - `grayscale`: グレースケール化（true 推奨）
  - `image_size`: 画像サイズ（既定 64）
  - `frame_stack`: フレームスタック数（既定 4 など、0/1 で無効）
- TrainConfig
  - `total_frames`: 学習総ステップ
  - `batch_size`, `collect_steps_per_iter`, `updates_per_collect`: 学習強度・頻度
  - `log_interval`: 標準出力へのログ頻度
  - `save_every_frames`: チェックポイント保存間隔（0 で無効）
  - `use_llm`, `llm_model`, `lambda_kl`: LLM 連携関連
- ModelConfig
  - `latent_dim`: 潜在次元
  - `obs_channels`: 入力チャンネル数（前処理から自動推定もされます）

参考（既定値・抜粋）
```python
from dataclasses import dataclass

@dataclass
class EnvConfig:
    name: str = "CrafterReward-v1"
    grayscale: bool = True
    image_size: int = 64
    frame_stack: int = 4

@dataclass
class TrainConfig:
    total_frames: int = 100000  # 例
    batch_size: int = 128
    collect_steps_per_iter: int = 200
    updates_per_collect: int = 50
    log_interval: int = 1000

@dataclass
class ModelConfig:
    latent_dim: int = 256
    obs_channels: int = 1
```

## チェックポイント
- 既定では `save_every_frames=0`（無効）。有効にすると `checkpoints/ckpt_step_XXXXX.pt` を保存します。
- 読み込みはコード内で `DreamerV3Agent.load(path)` を呼び出してください（CLI からの再開フローは必要に応じ追加してください）。

## LLM（任意）
LLM は API（Google Generative AI SDK）または CLI（gemini / gemini-cli）で呼び出せます。無効時はゼロ出力で学習が続きます。

### API モード（推奨。無料枠あり）
1) `LoRe/conf.py` で `TrainConfig.use_llm=True`
2) 環境変数を設定
```powershell
$Env:GEMINI_API_KEY = "<YOUR_KEY>"
# 任意: 既定モデルは gemini-2.5-flash-lite。上書きしたい場合
# $Env:GEMINI_MODEL = "gemini-2.5-flash-lite"
```
3) 実行
```powershell
python -m LoRe.main --total_frames 20000 --device cuda --log_dir runs/dreamer_crafter
```
メモ: 料金・無料枠（RPM/RPD/TPM）は Google の最新ドキュメントを参照してください。

### CLI モード（単発実行向け）
1) `LoRe/conf.py` で `TrainConfig.use_llm=True` と `TrainConfig.llm_use_cli=True`
2) `gemini`（または `gemini-cli`）が PATH にあることを確認
3) 学習時に観測ごとに単発呼び出し（内部で `echo "<prompt>" | gemini` → 失敗時 `gemini-cli prompt` を自動フォールバック）

CLI 例（参考）:
```bash
echo "リポジトリの最新コミット3件を要約" | gemini
gemini-cli prompt "why is the sky blue?"
```
内部プロンプトは JSON で `action_prior.logits`（A次元）, `confidence`, `features`（K次元）を要求します。

## ロギング
- `runs/<run_name>` に TensorBoard ログを保存。
- 標準出力は `log_interval` ごとにダイジェストを表示。

## トラブルシューティング
- Gym の非推奨警告: 無視可（内部で旧 API を吸収）
- TorchRL C++ バイナリ警告: 一部機能が使えない場合がありますが、本構成では致命的ではありません。
- Windows のメモリ/ページングエラー: 環境変数で回避
  ```powershell
  $Env:LORE_REPLAY_BACKEND = "tensor"
  ```
- メモリ不足: `replay_capacity`/`batch_size`/`frame_stack` を下げる
- ログが出ない: `log_interval` を小さくする（例: 200）
- LLM が無反応/遅い: タイムアウトを `LLMAdapterConfig.timeout_s` で短く設定（既定 2.5s）

## ディレクトリ構成
- `LoRe/envs/crafter_env.py`: Crafter→TorchRL 接続、前処理 Compose
- `LoRe/agents/dreamer_v3.py`: DreamerV3（world+actor-critic, TD(0), save/load）
- `LoRe/trainers/trainer.py`: 収集/更新、初期探索、LLM 配線、チェックポイント
- `LoRe/utils/replay.py`: リプレイ生成（Windows 向けバックエンド切替）
- `LoRe/utils/llm_adapter.py`: Gemini ラッパ（無効時はゼロ出力）
- `LoRe/main.py`: エントリポイント（観測から `obs_channels` 自動推定）

## よくある質問（FAQ）
- 速く回したい
  - 画像サイズやフレームスタックを下げる、`batch_size`/`updates_per_collect` を調整、`device=cuda` を使用
- 途中から再開したい
  - エージェントの `load(path)` を呼び出す処理を追加してください（将来的に CLI を追加予定）

---

ライセンスや引用が必要な場合は追記してください。
