いいね、その設計は“乗せ方”で化ける。Dreamer系を前提に、(A) リプレイ拡張, (B) 行動バイアス, (C) ワールドモデル内での計画支援の3レイヤで入れるのが安定です。まずは(B)→(A)→(C)の順で段階導入が現実的。

⸻

全体像（最小実装ロードマップ）
	1.	LLM→コード生成→サンドボックス実行
観測 o_t から LLM が 特徴量生成コード と 方策のヒント を出す（JSON）。コードは安全サンドボックスで実行し、数値特徴と行動スコアを得る。
	2.	Dreamer への入力拡張
o_t' = [o_t, f_t^LLM] をエンコーダへ。リプレイには o_t', a_t, r_t, ... に加え、LLMヒントもメタ情報として格納。
	3.	行動選択時の“LLM 事前分布”での軽いバイアス
俯瞰すると “RL as Inference” のノリ：RL方策 π に LLM 事前 ρ を掛ける/正則化する。
\[
\pi^\*(a|z) \propto \pi_{\text{RL}}(a|z)\,\rho(a|z)^{\alpha}
\]
もしくはアクター目的に \alpha\,\mathbb{E}_{\pi}[\log\rho(a|z)] を加える。
	4.	たまに LLM をプランナーとして使い、Dreamer の潜在で CEM を初期化
LLM が出すシーケンスを初期分布にして Dreamer のイマジネーションで評価→良い軌跡だけバッファへ重み付きで追記（Dyna風）。

⸻

(A) リプレイバッファ拡張の実務

バッファ1サンプル

(s_t_raw, s_t_llmfeat, a_t, r_t, s_{t+1}_raw, s_{t+1}_llmfeat, 
 llm: {prior_logits, plan_{t:t+H-1}, subgoal, confidence, code_id})

	•	s_t_llmfeat = f_t^LLM：LLM生成コードで計算した数値特徴（例：インジケータ、環境関数の出力、ゲームならタイルカウント等）。
	•	prior_logits：離散行動の生logit、連続なら平均・分散かスコア。
	•	confidence：LLM自己評価。後述の温度/重み調整に使う。
	•	失敗（実行エラー/NaN）は mask=0 で学習時に無視。

ポイント
	•	LLM特徴は別エンコーダに通してから結合（late-fusionが安定）。
	•	生成コードは決定的で再現可能に（固定乱数、バージョン付与、code_id の保存）。

⸻

(B) 行動判断に LLM を効かせる3パターン
	1.	KL正則化（推奨の第一歩）
Dreamerのアクター損失に LLM事前を加える：
J_{\text{actor}} = \mathbb{E}\big[\hat{Q}(z_t,a_t)\big]
\;-\; \lambda\,\mathrm{KL}\!\left(\pi(\cdot|z_t)\,\|\,\rho(\cdot|z_t) \right)

	•	\rho は LLM 由来の行動分布。\lambda は小さく始めてスケジュールで調整。
	•	連続行動なら LLM をガウス事前に投影（平均・分散を出す or 小さなMLPで写像）。

	2.	アドバンテージ加点（ボーナス整形）
\tilde{A}_t = A_t + \beta\,b_t^{\text{LLM}}

	•	b_t^{\text{LLM}}：LLM が出すその行動のヒューリスティック・スコア（-1〜1などに正規化）。
	•	直感的で実装楽。過信防止に \beta は不確実性連動（後述）。

	3.	混合方策（ゲーティング）
\pi_{\text{mix}} = (1-g_t)\,\pi_{\text{RL}} + g_t\,\rho

	•	g_t \in [0,1] は不確実性（Criticの分散やensemble不一致、LLM信頼度）から学習。
	•	長期的には蒸留で \pi_{\text{mix}} → \pi_{\text{RL}} へ吸収し、LLMコールを減らす。

⸻

(C) ワールドモデル×LLMプランニング
	•	Dreamerの潜在 z_t 上で CEM/MPPI を回す時、初期サンプルやノイズ共分散を LLM提案でブートストラップ。
	•	LLMがサブゴール（自然言語or構造化）を出す → サブゴールエンコーダで潜在目標 z_g へ写像 → Goal-conditioned Dreamer に。

疑似コード（CEM初期化）

mean, cov = llm_proposal_to_gaussian(plan_H)  # ρ 由来
for it in range(K):
    A = sample_action_sequences(mean, cov, N)
    R = evaluate_in_world_model(z_t, A)       # Dreamer imagination
    elite = select_top(A, R, top_p=0.1)
    mean, cov = refit_gaussian(elite)
a_t = mean[0]


⸻

LLM I/O 仕様（堅牢化の肝）

プロンプトの期待出力（JSON）

{
  "features_code": "def compute_features(obs): ... return {'k1': v1, 'k2': v2}",
  "action_prior": {"type": "discrete", "logits": [0.2, -0.1, 0.0]},
  "plan": {"horizon": 8, "actions": [ ... ]},
  "subgoal": {"kind": "numeric", "value": [0.3, -0.1]},
  "confidence": 0.62,
  "safety_notes": ["never call network", "no file writes"]
}

	•	型/範囲検証→ユニットテスト→タイムアウト実行→NaNガード。
	•	実行は完全サンドボックス（権限剥奪、CPU/メモリ/時間制限、I/O遮断）。

⸻

Dreamer への組み込み（最小差分）
	•	Encoder: enc_raw(o_t) と enc_llm(f_t) を concat → z_t。
	•	Actor loss 追加入力: LLM 事前分布 ρ(a|z_t) を別頭で出し（あるいは外部入力）、上記 KL/ボーナスを追加。
	•	Critic は素でOK（ただし LLM特徴が入るので overfittingを防ぐため weight decay/Dropout 少々）。

⸻

不確実性での重み付け
	•	RL側の不確実性: Criticのアンサンブル分散、価値のAleatoric/Epistemic推定。
	•	LLM側の不確実性: confidence、過去Nステップでの実績精度（ヒントに従った時のAUC/Sharpe/スコア）。
	•	重み例：\alpha = \alpha_0 \cdot \text{conf}{\text{LLM}} \cdot \sigma{\text{RL}}（RLが不確か＆LLMが自信ある時だけ強く効かせる）。

⸻

学習ループ（ざっくり擬似コード）

for step in range(T):
    o = env.obs()
    f_llm, rho, plan = llm_tool(o)          # キャッシュ＆間引き推奨
    a_rl = actor.sample(o, f_llm)
    a = mix_action(a_rl, rho, gate(o))      # KL/ボーナス/ゲートのどれか
    o2, r, d, info = env.step(a)

    replay.add(o, f_llm, a, r, o2, rho, plan, conf=rho.conf)
    if step % update_every == 0:
        batch = replay.sample(B)
        z = world.encode(batch.o, batch.f_llm)
        imagine = world.rollout(z)
        actor_loss = -Q(z,a) + alpha*log_rho(a|z)    # いずれかの形
        critic_loss = ...
        optimize(actor_loss + critic_loss)

    if step % distill_every == 0:
        # π_mix を模倣して π_RL を蒸留（コール削減）
        distill_update()


⸻

まずこれから（現実的スターター）
	1.	LLM特徴だけを追加（移動平均や差分など、コード実行は軽いやつ）。
	2.	KL正則化で LLM 事前を弱めに（\lambda\!\approx\!0.01 から）。
	3.	ゲートは Critic分散とconfidenceの AND 条件でON。
	4.	成果が見えたら CEM初期化を試す。
	5.	週次で蒸留、月次でアブレーション（LLM無効/特徴だけ/事前だけ/両方）。

⸻

典型トラブルと対策
	•	リーク/過信：LLMが短期の見かけ利得を過剰に押す → \lambda,\beta を自動調整（バンディットでメタ最適化）。
	•	コード脆弱：テスト＋フェイルセーフ（ゼロ埋め＆mask）。
	•	概念ドリフト：confidenceを時系列で再校正、古いコードは勝率低下で自動淘汰。
	•	推論コスト：LLMコールはスパース化（イベント駆動/不確実時のみ）、蒸留で削減。

⸻

金融（離散：買う/保持/売る）に当てるなら
	•	f_t^LLM：市場体制判別（レジーム）、ボラ急変度、ニューススコア等。
	•	ρ(a|s)：LLMが出す「いまは売り寄り」分布。
	•	実運用では \alpha をボラ高で強める、コスト/スリッページを r に内生化。

⸻

必要なら、あなたの環境（観測・行動の型、Dreamer実装）に合わせた具体的なクラス設計と損失式まで落とし込みます。使うフレームワーク（PyTorch/Haiku）教えてくれたら、最初のPRレベルの骨組み書くよ。

できる。**TorchRL + Dreamer（PyTorch）**なら、
	1.	リプレイバッファにLLM由来の情報をそのまま追加、
	2.	アクター損失にLLMの事前分布(KL)やボーナスを足す、
の2点を小改造で入れられる。

⸻

1) リプレイバッファのLLM拡張

TorchRLはTensorDictで好きなキーを増やせる。収集側でLLMを呼び、その出力をTensorDictに突っ込むだけ。

収集ループ（概略）

td = env.reset()  # td["observation"]などが入ってる

# たとえば不確実時のみ呼ぶ
if need_llm(td):
    llm_out = llm_adapter(td["observation"])  # ユーザ実装: prior_logits(17), conf, features(dict)

    # リプレイに入れるキーを追加
    td.set("llm_prior_logits", llm_out["prior_logits"])   # [B, 17]
    td.set("llm_confidence",   llm_out["confidence"])     # [B, 1]
    td.set("llm_mask",         torch.ones_like(llm_out["confidence"]))  # 失敗時0に
    td.set("llm_features",     torch.tensor(llm_out["features_vec"]))   # 例: [B, K]
else:
    td.set("llm_prior_logits", torch.zeros(B, 17))
    td.set("llm_confidence",   torch.zeros(B, 1))
    td.set("llm_mask",         torch.zeros(B, 1))
    td.set("llm_features",     torch.zeros(B, K))

# 行動サンプリング → 環境step
td = actor(td)        # Dreamerの方策でa_tをセット
td = env.step(td)     # 次状態はtd["next"][...]

# そのままReplayへ
replay.add(td)

ポイント
	•	追加キー例：
llm_prior_logits（離散17次元）, llm_confidence（0〜1）, llm_mask（0/1）, llm_features（任意次元）, llm_plan（必要なら短いHorizonの行動列）。
	•	失敗（タイムアウト/NaN）は llm_mask=0 にして学習時に無視。
	•	TensorDictReplayBuffer(storage=LazyMemmapStorage(N)) は追加キーもそのまま保存できる。

LLM特徴の取り込み（late-fusion 推奨）
	•	Dreamerのエンコーダに別のMLPを1個足してz = concat(z_raw, enc_llm(llm_features))にするだけ。
ここは後回しでもOK（最初は行動事前だけでも効果が出ることが多い）。

⸻

2) 行動判断へのLLMバイアス（損失に1〜2行）

Dreamerの学習ステップでアクター損失を取り出す部分に、LLM事前を混ぜる。

A. KL正則化（いちばん素直）

# 既存: actor_loss = loss_dreamer["actor_loss"]
pi_dist = loss_dreamer["policy_dist"]          # Categorical(logits=π_logit) を想定
rho_dist = Categorical(logits=td["llm_prior_logits"])

kl = torch.distributions.kl.kl_divergence(pi_dist, rho_dist)  # [B]
# 不確実性に応じて重みを変える（例：信頼度×マスク）
w = (td["llm_confidence"].squeeze(-1) * td["llm_mask"].squeeze(-1)).detach()

actor_loss = actor_loss + (lambda_kl * (kl * w).mean())

	•	lambda_klは0.01くらいから。
	•	連続行動ならNormal/MultivariateNormalで同様。

B. アドバンテージ加点（ボーナス整形）

LLMが各行動の“好み”スコアをくれられるなら、logρ(a|s)をボーナスに：

logp_prior = rho_dist.log_prob(td["action"].squeeze(-1))  # [B]
actor_loss = actor_loss - beta_bonus * (w * logp_prior).mean()

C. 混合方策（ゲート）

推論時のみ：

with torch.no_grad():
    g = gate(td)  # 0〜1（例：Critic ensembleの分散やconfから計算）
    pi = pi_dist.probs
    rho = rho_dist.probs
    probs_mix = (1-g) * pi + g * rho
    a = Categorical(probs=probs_mix).sample()
td.set("action", a.unsqueeze(-1))

学習は通常のDreamer、推論を混合にする→安定＆実装ラク。後で蒸留でπに吸収。

⸻

3) 仕様メモ（Crafter前提：離散17）
	•	llm_prior_logits: 形状 [B, 17]。未提供時は0ベクトル＋llm_mask=0。
	•	llm_confidence: [B,1]（0〜1）。スケジューリングに使う。
	•	llm_features: [B, K]（任意）。最初は未使用でもよい。
	•	収集時のLLM呼び出しは間引く（Nステップに1回 or エピソード序盤のみ or 不確実時のみ）。
	•	不確実時判定例：Criticアンサンブルの分散が閾値超え、報酬予測損失が大きい、など。

⸻

4) 最小改造の入れどころ（目安）
	•	Collector：環境step前後でllm_adapterを呼ぶ1ブロック＋td.set(...)を追加。
	•	Loss/Trainer：アクター更新の直前でkl_divergence(...)またはlog_prob項を1〜2行足す。
	•	（任意）Encoder：llm_featuresを小MLPで埋め込み→zにconcat。

⸻

5) 失敗しやすいポイントと回避
	•	形状ずれ：Categoricalは[B]/[B,1]で挙動が変わる。squeeze(-1)で統一。
	•	事前の暴走：lambda_kl/beta_bonusは小さく開始、wでマスク×信頼度。
	•	コスト：LLMはイベント駆動（不確実/サブゴール切替時のみ）。キャッシュ必須。
	•	リーク：LLMコード実行はサンドボックス＋NaNガード。不可ならllm_mask=0に。

⸻

6) まずの動かし方（順番）
	1.	TorchRL Dreamer + Crafterを素で学習
	2.	収集にllm_prior_logitsだけ追加（最初はダミー乱数でもOK）→配線確認
	3.	KL正則化を1行追加（λ=0.01）
	4.	不確実時トリガ＆信頼度ウェイトw導入
	5.	余力があればllm_featuresのlate-fusion

必要なら、使ってるTorchRLのサンプル（ファイル名）教えて。そのファイルの具体的な関数名・行番号に合わせた差分パッチ書く。