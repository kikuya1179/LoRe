# 方針まとめ（TL;DR）

* **経路を完全分離**：①「リプレイ拡張（LLMコード→特徴/疑似報酬/サブゴール等を“データ化”してRBに保存）」と、②「意思決定時の介入（LLM由来のprior/マスク/オプションでπを修正）」は別ラインで設計。
* **コードは“実行結果だけ”を学習に使う**：生コードはハッシュで参照し、RBには**軽量な出力**（logits/feature/subgoal/shaped\_r/confidence 等）だけを入れる。
* **安全な実行**：**制限付きDSL or サンドボックス化Python**で走らせ、タイムアウト・リソース制限・AST検査・I/O禁止を徹底。
* **コスト対策**：新奇度/停滞/TD誤差で**呼び出しを間引く＋キャッシュ**。介入は蒸留して**徐々にα→0**に。

---

# 0) 用語

* `LLM-code`：LLMが生成する短いコード（DSL or 制限Python）。
* `CodeOut`：LLM-codeの**実行結果**（JSON相当）。
* `RB`：Replay Buffer。
* `π_actor`：通常のポリシー（Dreamerのactor）。
* `π_llm`：LLMから得られる**行動事前**（logits もしくは action-mask など）。

---

# 1) リプレイバッファ拡張（「コードをぶち込む」側）

## 1.1 役割と狙い

* LLM-code で**ラベル・特徴・潜在報酬・サブゴール**を“後付け”して、学習効率・探索効率を上げる。
* 直接の行動介入ではなく、**データ拡張/教師信号強化**のラインとして独立運用。

## 1.2 LLMコードの入出力仕様（推奨）

**入力（環境依存の観測）**

* `obs`：画像（リサイズ済み）、数値状態、recent actions/rewards（オプション）。
* `context`：タスク定義、禁止事項、行動空間、成功条件など（プロンプト側で定型化）。

**出力（CodeOut; JSON）**

```json
{
  "features":  [f1, f2, ... fK],          // 追加特徴（例: goal_distance, risk_score）
  "subgoal":   {"id":"cook_wood", "tau":8},// サブゴール提案（継続長τ）
  "r_shaped":  0.12,                       // 0に近い微小 shaping（ポテンシャル型推奨）
  "policy":    {"logits":[...], "temp":1.0}, // 低次元actionなら logits でOK
  "mask":      [0,1,1,1,0],                // 違法/無意味アクションのマスク
  "confidence":0.73,
  "notes":     "make_tool -> then mine stone"
}
```

* **重要**：`r_shaped` は**ポテンシャル型**（`F(s')-F(s)`）で設計し、最適方策不変性を極力守る。`features` は**正規化**前提（平均0±1程度）。

## 1.3 コードの形式：DSL優先 or 制限Python

* **DSL案（推奨）**：

  * 許可演算子・APIを限定（算術/比較/簡単な畳み込み/領域抽出程度）。
  * 例：`goal("craft_stone_pickaxe")`, `count("wood")`, `has("workbench")` などの**高レベル述語**を用意。
  * 実行器は自作で安全（ASTが単純、TLE/メモリ制限も容易）。
* **制限Python案**：

  * `RestrictedPython` + **サブプロセス**実行 + `seccomp/rlimit` + **I/O/ネット/ファイル禁止** + 200msタイムアウト。
  * 許可importは**皆無**、与えるbuiltinsも最小。
  * 返り値は`CodeOut`型に**厳密スキーマ**チェック（pydantic等）。

## 1.4 実行とキャッシュ

* **キー**：`state_hash = H(encoder(obs), discretized_stats)`（世界モデルの潜在`z`を量子化してもOK）。
* **キャッシュ**：`(model_id, prompt_id, state_hash)`→`CodeOut`。
* **実行頻度**：

  * `novelty > τ_novel` または `TD-error > τ_td` または `学習停滞信号` のときのみ。
  * バッチ実行：収集サイクルごとに「要解析フラグ付き遷移」を束ねてLLMへ。

## 1.5 RBスキーマ拡張（メモリを食い過ぎない）

RB 本体には**軽量値のみ**：

```python
class RBExtra(NamedTuple):
    code_id: int32        # コード本文は外部KVに置く（必要時のみ取得）
    conf: float16
    r_shaped: float16
    subgoal_id: int16
    subgoal_tau: uint8
    feat: float16[K]      # 小さめK(8~32)
    logits: float16[A]    # Aが大なら保存しない/圧縮 or top-kのみ
    mask_bits: uint64     # A<=64ならbitmaskで圧縮
```

* **コード原文**は`code_id -> blob`を**別ストア**（LMDB/SQLite/FS）に。RBは参照だけ。

## 1.6 学習への使い方

* **報酬整形**：`r_total = r_env + β * r_shaped`（β小さめ & クリップ）。
* **優先度サンプリング**：`prio = α*TDerr + (1-α)*(1-conf)`（低confや高TDを優先）。
* **HER/ラベル補完**：`subgoal`を用いて「達成後ラベル」や「未達ペナルティ」を後付け。
* **蒸留用タグ**：`has_llm=True`を保存し、**学習時にKL(π\_actor || π\_llm) を“オフライン”にかける**（本番のLLM頻度を下げる布石）。

---

# 2) RLの意思決定にLLMを作用（オンライン介入）

## 2.1 介入のレベル（選べる三段階）

**L1: 行動priorミックス（簡単で強い）**

* 公式：`log π_final(a|s) = log π_actor(a|s) + α(s) * log π_llm(a|s)`
* `α(s) = α0 * g(confidence, novelty, stage)`（例：`α0∈[0,1]`、conf高＋新奇高で強め）。
* 実装は**actor出力logitsへ加算**→softmax。

**L2: 行動マスク（安全/ルール注入）**

* `mask[a]=0` を `-∞` ロジット加算で実現。
* 明確な**違法行動/明確に無意味行動**を弾く。

**L3: オプション/ハイレベル（数ステップ固定）**

* `subgoal(id, τ)` を採択したら、**低レベルは actor に任せる**（hintとして features をconcat）。
* 期間`τ`は**早期終了条件**（失敗/達成）で中断可。
* これは Dreamer の**想像ロールアウト**にも自然に乗る（`features`を潜在にconcat）。

## 2.2 学習損失への組み込み

* **オンライン蒸留**（介入フレームのみ）
  `L_distill = λ * KL(π_actor || π_llm)`（λは小さく。trust-region的に過信しない）。
* **行動価値の正則化**
  `L_reg = μ * E_a[π_llm(a|s) * (-Q(a,s))]` で“悪手に罰”をかける等、軽めの補助。
* **αアニーリング**
  反復が進むほど `α→0` に落とし、方策を**自立**させる。

## 2.3 実装ポイント（Dreamer系）

* **concat特徴**：`h_t = enc(s_t)` に `feat_llm` を `LayerNorm`後でconcat。
* **logits加算**：`actor(h_t)` のlogitsへ `α * logits_llm` を加算（maskは `-1e9`）
* **学習時**：介入stepにだけ `KL(π_actor||π_llm)` を載せる（すべてに載せない）。
* **記録**：`used_llm`, `alpha`, `conf`, `hit_cache` をTensorBoardに。

---

# 3) 失敗時の扱い・安全

* **タイムアウト**：コード実行は 200ms\~500ms/件、超えたら**スキップ**。
* **バリデーション**：出力スキーマ不一致→discard（`treat_as_true`のような全面Trueフォールバックは厳禁）。
* **I/O禁止**：ネット/ファイル/環境変数アクセスは**完全遮断**。
* **サイズ上限**：コード 1KB、CodeOut JSON 1KB 程度に制限。
* **ロールバック**：クラッシュ時は `β=0, α=0`（完全通常RL）に自動退避。

---

# 4) コスト最適化と呼び出し戦略

* **新奇度トリガ**：`kNN(z_t)`距離や**予測誤差**が閾値超えたら呼ぶ。
* **停滞トリガ**：評価成功率/returnが `Δ<ε` の区間長が続いたら頻度↑。
* **TD誤差選別**：上位p%の遷移のみ LLM 処理。
* **キャッシュ**：`(model_id, prompt_id, state_hash)` をキーに**高ヒット率**を狙う。
* **バッチ**：収集バッファから**まとめて送信**し、API往復を削減。

---

# 5) LoRe への落とし込み（変更箇所のイメージ）

* `llm/adapter.py`

  * `run_llm(obs_batch) -> List[CodeOut]`（CLI/API/キャッシュ/制限実行器を内包）
* `llm/sandbox.py`

  * DSL実行器 or 制限Python実行器（ASTチェック/timeout/rlimit）
* `replay/buffer.py`

  * `RBExtra` フィールド追加、`code_kv` ストア（LMDB/SQLite）
* `trainer.py`

  * 収集フェーズ末尾で「要解析遷移」を `run_llm` に回す & `RB` を**後付け更新**
  * 学習フェーズで `r_shaped`/`feat`/`logits`/`mask` を取り込み
  * 介入stepに `KL_distill` と α制御を適用
* `agent/actor.py`

  * `forward(h)` に `feat_llm` concat（LayerNorm）
  * `logits += α*logits_llm; logits[mask==0]-=1e9`
* `eval/metrics.py`

  * `used_llm/alpha/conf/cache_hit/llm_calls_per_kframe` を記録
  * Crafterの成功率/アチーブメントをA/B比較

---

# 6) 参考プロンプト設計（コード生成型）

* **システム**：「あなたはCrafterの攻略プランナー。以下の観測から、短い“安全なDSL”で特徴/サブゴール/軽い報酬/行動priorを返してください。」
* **ツール仕様**（DSLリファレンス）を**先に固定**し、**返却は JSON + `code`** の二部構成に限定。
* **例示**：2\~3ケースの**成功/失敗**のミニ例を含める（過学習を避ける短さで）。
* **出力要件**：`features[-3..3]` 範囲に正規化、`r_shaped∈[-0.2,0.2]`、`mask`はA次元厳密長、など**境界条件を明記**。

---

# 7) 具体的な疑似コード

**収集→LLM後付け（バッチ）**

```python
traj = collector.rollout(env, policy=actor)  # 通常収集
pending = select_for_llm(traj, td_error, novelty, plateau)
if len(pending):
    codeouts = llm_adapter.run_llm(pending.obs_batch)   # キャッシュ＋制限実行
    for t, co in zip(pending.indices, codeouts):
        rb.update_extra(t, to_RBExtra(co))              # 後付けで埋める
```

**学習（actor更新）**

```python
h = world.encode(obs)                     # Dreamer潜在
h = concat(h, norm(co.features))          # feat concat
logits = actor(h)
if co.policy is not None:
    logits += alpha(s, co.conf) * co.policy.logits
if co.mask is not None:
    logits[co.mask==0] -= 1e9
pi = softmax(logits)

# 蒸留は“介入stepのみ”
loss = rl_loss(pi, Q) + lambda_kl * KL(pi || softmax(co.policy.logits))
```

---

# 8) 評価計画（最短で効果を見る）

1. **A/B**：`Baseline`（α=0, β=0） vs `RBのみ`（β>0） vs `Decisionのみ`（α>0） vs `両方`。
2. **指標**：

   * サンプル効率（成功率がX%到達までのフレーム数）
   * LLMコスト（calls/k frames, cache hit率）
   * 介入比率（used\_llm%）/ 蒸留後の α 低下による**自立度**
3. **アブレーション**：`maskのみ`/`logitsのみ`/`featのみ`/`r_shapedのみ`。

---

# 9) 最小導入ステップ（今日から入れる順）

1. **RB拡張**（`RBExtra` + LMDB/FSの`code_kv`）
2. **LLMアダプタ**（キャッシュ・バッチ・制限実行器・スキーマ検証）
3. **actorへのlogits加算 & mask適用**（αは定数から開始）
4. **`r_shaped`足し込み + 優先度サンプリング更新**
5. **蒸留（介入stepのみKL）** → **αアニーリング**
6. **メトリクス整備**（成功率/コスト/α/conf）

---

必要なら、この設計をそのまま\*\*パッチ雛形（ファイル構成・型・pydanticスキーマ・テスト）\*\*に落として出すよ。`DSL`で行くか`制限Python`で行くかだけ決めてくれれば、最短で実装切り出す。
