# LoRe

## セットアップ

- 推奨 Python: 3.10
- Windows PowerShell 手順は `LoRe/INSTALL_WINDOWS.txt` を参照。
- 依存インストール:
  ```bash
  pip install -r LoRe/requirements.txt
  ```

## DreamerV3 + Crafter + LLM(Gemini)

- エージェントは `DreamerV3Agent` に統一。
- `TrainConfig.use_llm=True` で LLM 事前分布と KL 正則化を有効化（API キー `GEMINI_API_KEY`）。


