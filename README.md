# 大規模言語モデル（LLM）講義 - Databricks ハンズオン

大学などの講義（90分）で使用する、Databricks環境での大規模言語モデル（LLM）の実践的なハンズオン教材です。Foundation Model APIからオープンソースモデルのファインチューニング、MLflowによる評価・管理まで、LLMの基礎から本番運用まで体系的に学習できます。

## 📚 概要

本教材は、以下のトピックをカバーする6つのExerciseで構成されています：

1. **Exercise 1**: Chat Completion APIの基本
2. **Exercise 2**: Structured Outputsによるデータ抽出
3. **Exercise 3**: Function Callingの基礎
4. **Exercise 4**: HuggingFaceモデルのローカル実行
5. **Exercise 5**: LoRAファインチューニング
6. **Exercise 6**: MLflowによるモデル評価と実験管理 ⭐ **NEW**

## 🎯 学習目標

- Databricks Foundation Model APIの使用方法を習得
- Structured OutputsとFunction Callingの実装
- オープンソースモデル（Gemma 3 270M）の活用
- LoRAを使ったパラメータ効率的なファインチューニング
- **MLflowによる実験トラッキングとモデル管理** ⭐ **NEW**
- **LLM評価指標とベストプラクティス** ⭐ **NEW**
- **Model Registryを使った本番デプロイ** ⭐ **NEW**
- 実務で使えるLLMアプリケーション開発スキルの獲得

## 📋 前提条件

### 必須
- Databricks環境へのアクセス（Free Edition以上）
- Python基礎知識
- 機械学習の基本的な理解

### 推奨
- GPU対応クラスタ（Exercise 5, 6で使用）
- MLflowの基礎知識（Exercise 6）

## 🚀 セットアップ

### 1. Databricks環境の準備

**Databricks Free Edition**
```
1. https://www.databricks.com/try-databricks にアクセス
2. アカウントを作成
3. Notebookを新規作成
```

### 2. 必要なライブラリのインストール

各ExerciseのNotebook冒頭で以下が自動実行されます：

```
# Exercise 1-3
%pip install --upgrade databricks-sdk openai pydantic

# Exercise 4-5
%pip install --upgrade transformers datasets accelerate peft trl bitsandbytes sentencepiece

# Exercise 6
%pip install --upgrade transformers datasets evaluate rouge-score bert-score sacrebleu nltk peft torch mlflow
```

### 3. HuggingFace Tokenの設定（Exercise 4-6）

Gemma 3モデルを使用するには、HuggingFaceのトークンが必要です：

1. [HuggingFace](https://huggingface.co/)でアカウント作成
2. [Gemma 3 270M-IT](https://huggingface.co/google/gemma-3-270m-it)のライセンスに同意
3. [Tokensページ](https://huggingface.co/settings/tokens)でRead権限のトークンを生成
4. Notebookで認証:
```
from huggingface_hub import login
login(token="your_token_here")
```

## 📖 Exercise概要

### Exercise 1: Chat Completion APIの基本（8分）

**目的**: Databricks Foundation Model APIの基本的な使い方を理解する

**内容**:
- WorkspaceClientの初期化
- シンプルな質問応答
- システムプロンプトの効果
- Temperatureパラメータの調整
- マルチターン会話の実装
- トークン使用量の監視

**使用モデル**: `databricks-meta-llama-3-3-70b-instruct`

**主な学習ポイント**:
```
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {"role": "user", "content": "機械学習とは何ですか？"}
    ],
    max_tokens=256,
    temperature=0.7
)
```

### Exercise 2: Structured Outputsによるデータ抽出（12分 + おまけ5分）

**目的**: 非構造化テキストから構造化データを抽出する実践的スキルを習得

**内容**:
- Pydanticによるスキーマ定義
- JSON Schemaを使った構造化出力
- 顧客レビューの自動分析
- DataFrameへの変換とビジネス分析
- バリデーションと品質管理
- **おまけ**: Databricksバッチ推論（ai_query、Pandas UDF、Streaming）

**ビジネスユースケース**: Eコマース顧客レビュー分析

**主な学習ポイント**:
```
from pydantic import BaseModel, Field
from typing import Literal

class ReviewAnalysis(BaseModel):
    product_name: str = Field(description="製品名")
    rating: int = Field(description="評価（1-5の整数）")
    sentiment: Literal["positive", "negative", "neutral"]

response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "review_analysis",
            "schema": ReviewAnalysis.model_json_schema(),
            "strict": True
        }
    }
)
```

### Exercise 3: Function Callingの基礎（10分）

**目的**: LLMが外部ツールを呼び出す仕組みを理解し、AIエージェントの基盤を構築

**内容**:
- ツール定義（天気情報、在庫確認、配送料金計算）
- マルチターン会話でのツール実行フロー
- 複数ツールからの自動選択
- tool_choiceパラメータの制御
- 次回のAIエージェント講義への橋渡し

**ビジネスユースケース**: カスタマーサポートボット

**主な学習ポイント**:
```
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "指定された都市の現在の天気情報を取得します",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[{"role": "user", "content": "東京の天気を教えて"}],
    tools=tools,
    tool_choice="auto"
)
```

### Exercise 4: HuggingFaceモデルのローカル実行（15分）

**目的**: Foundation Model API以外の選択肢として、オープンソースモデルを直接使用

**内容**:
- HuggingFace Hubからのモデルダウンロード
- `apply_chat_template`を使った正しいチャット形式
- Pipeline APIとlow-level APIの使い分け
- マルチターン会話の実装
- 様々なタスク（質問応答、要約、分類）での評価
- パフォーマンス測定とメモリ管理
- バッチ推論

**使用モデル**: [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)

**主な学習ポイント**:
```
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 方法1: Low-level API
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [{"role": "user", "content": "自己紹介してください"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)

# 方法2: Pipeline API
pipe = pipeline("text-generation", model="google/gemma-3-270m-it")
result = pipe(messages)
```

### Exercise 5: LoRAファインチューニング（20分 + トレーニング15-30分）

**目的**: LoRAを使った効率的なファインチューニングを実践

**内容**:
- データセットの準備とチャット形式への変換
- 4-bit量子化によるメモリ削減
- LoRA設定とPEFTライブラリの使用
- SFTTrainerによるファインチューニング
- ファインチューニング前後の性能比較
- モデルの保存とロード

**使用データセット**: [bbz662bbz/databricks-dolly-15k-ja-gozarinnemon](https://huggingface.co/datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon)
- Databricks Dolly 15kの日本語訳版
- 回答の語尾が「ござる」口調（効果が視覚的に確認しやすい）

**主な学習ポイント**:
```
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# LoRA設定
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# トレーニング
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    peft_config=lora_config
)

trainer.train()
```

**パラメータ効率**:
- 全パラメータファインチューニング: 270M（100%）
- LoRA（r=16）: 約0.5M（0.2%）
- **削減率**: 99.8%

### Exercise 6: MLflowによるモデル評価と実験管理（20分）⭐ **NEW**

**目的**: MLflowを使った実験トラッキング、モデル評価、本番デプロイを学ぶ

**内容**:
- MLflowによる実験の自動トラッキング
- 評価指標（BLEU、ROUGE、BERTScore）の記録
- MLflow Datasetsによるデータ管理
- カスタム評価指標の定義
- LLM-as-a-Judgeの実装とトラッキング
- Model Registryへの登録とステージング
- Model Servingへのデプロイ（デモ）
- 総合評価レポートの自動生成

**Databricks特有の機能**:
- MLflow UIでの実験比較
- Unity Catalogとの統合
- Lakehouse Monitoringへの接続

**主な学習ポイント**:
```
import mlflow
import mlflow.transformers

# MLflow実験の設定
mlflow.set_experiment("/Users/your-name/llm-evaluation")

# 評価実行とトラッキング
with mlflow.start_run(run_name="finetuned-model-eval") as run:
    # データセットをログ
    mlflow.log_input(dataset_source, context="evaluation")
    
    # パラメータをログ
    mlflow.log_param("model_name", model_id)
    mlflow.log_param("num_parameters", model.num_parameters())
    
    # メトリクスをログ
    mlflow.log_metric("bleu_score", bleu_result['score'])
    mlflow.log_metric("rouge1_score", rouge_result['rouge1'])
    mlflow.log_metric("bertscore_f1", f1_score)
    
    # モデルを登録
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        registered_model_name="gemma-3-270m-finetuned"
    )
```

**評価指標**:
1. **自動評価**
   - BLEU: n-gramベースの精度測定
   - ROUGE: 再現率ベースの要約評価
   - BERTScore: 意味的類似度の測定

2. **カスタム評価**
   - スタイル一貫性（ござる口調使用率）
   - 応答長の分析
   - 推論速度の測定

3. **LLM-as-a-Judge**
   - 強力なLLMによる品質評価
   - 人間の判断により近い評価
   - 説明可能性の高い評価

**Model Registryとデプロイ**:
```
from mlflow.tracking import MlflowClient

client = MlflowClient()

# モデルをStagingに昇格
client.transition_model_version_stage(
    name="gemma-3-270m-finetuned",
    version="1",
    stage="Staging"
)

# 性能基準を満たせばProductionに昇格
if meets_production_criteria:
    client.transition_model_version_stage(
        name="gemma-3-270m-finetuned",
        version="1",
        stage="Production"
    )
```

## 💡 推奨学習順序

### 講義内（90分）
1. **Exercise 1 → 2 → 3**: Foundation Model APIの基礎から応用（30分）
2. **Exercise 4**: オープンソースモデルの理解（15分）
3. **Exercise 5**: ファインチューニング実行（演習として開始）

### 講義後の発展学習
4. **Exercise 5**: ファインチューニング完了の確認
5. **Exercise 6**: MLflowによる評価と本番デプロイ（重要）⭐

## 🏗️ アーキテクチャ概要

```
┌───────────────────────────────────────────────────────────────┐
│                   Databricks Workspace                        │
│                                                               │
│  ┌──────────────────┐   ┌─────────────────────┐             │
│  │ Foundation Model │   │ HuggingFace Models  │             │
│  │ API              │   │ (Gemma 3 270M)      │             │
│  │ - Llama 3.3 70B  │   │ - Direct inference  │             │
│  │ - Gemini         │   │ - LoRA fine-tuning  │             │
│  │ - Qwen           │   └─────────────────────┘             │
│  └──────────────────┘                                         │
│         ↓                        ↓                            │
│  ┌──────────────────────────────────────────────┐            │
│  │ Applications                                  │            │
│  │ - Structured Output (Review Analysis)        │            │
│  │ - Function Calling (Customer Support)        │            │
│  │ - Custom Domain (Fine-tuned models)          │            │
│  └──────────────────────────────────────────────┘            │
│                                                               │
│  ┌──────────────────────────────────────────────┐            │
│  │ MLflow (Experiment Management) ⭐ NEW         │            │
│  │ - Experiment Tracking                         │            │
│  │ - Model Registry                              │            │
│  │ - Model Serving                               │            │
│  └──────────────────────────────────────────────┘            │
│                                                               │
│  ┌──────────────────────────────────────────────┐            │
│  │ Data Processing                               │            │
│  │ - ai_query() for batch inference             │            │
│  │ - Pandas UDF for complex logic               │            │
│  │ - Structured Streaming for real-time         │            │
│  └──────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────┘
```

## 📊 本番事例（Exercise 2で紹介）

1. **UiPath - 企業文書の構造化抽出**
   - DocPath: 請求書、領収書、税務書類の自動処理
   - Structured Outputsによる位置情報付き抽出
   - 自動化率200%向上

2. **Morgan Stanley - 顧客ミーティング議事録の自動構造化**
   - AI @ Morgan Stanley Debrief
   - 98%のFinancial Advisorチームが採用
   - 1ミーティングあたり30分の時間削減

3. **Klarna - カスタマーサポート自動化**
   - 1億5000万ユーザー、1日250万件の取引処理
   - 700人分のフルタイムエージェント業務を代替
   - 平均解決時間を80%削減

## 🛠️ トラブルシューティング

### Exercise 2: "Invalid JSON schema - integer types do not support minimum"

**原因**: Databricks strict モードでは `ge`/`le` 制約がサポートされていない

**解決策**: 
```
# ❌ 使用不可
rating: int = Field(ge=1, le=5)

# ✅ 正しい方法
rating: int = Field(description="評価（1-5の整数）")
```

### Exercise 2 バッチ推論: "default auth: cannot configure default credentials"

**原因**: Pandas UDF内でWorkspaceClientが認証できない

**解決策**: ドライバーで認証情報を取得し、ブロードキャスト変数で配布
```
w = WorkspaceClient()
token = w.config.token
broadcast_token = spark.sparkContext.broadcast(token)

# UDF内で使用
token = broadcast_token.value
client = OpenAI(api_key=token, base_url=...)
```

### Exercise 4-5: GPU not available

**解決策**: 
1. Serverless GPUを使用（推奨）
2. GPU対応クラスタ（g4dn.xlarge以上）を起動
3. Databricks Runtime 14.3 ML以降を選択

### Exercise 5: Out of Memory Error

**解決策**:
```
# バッチサイズを削減
per_device_train_batch_size=2  # 4 → 2

# 勾配累積ステップを増加（実効バッチサイズ維持）
gradient_accumulation_steps=8  # 4 → 8

# 勾配チェックポイントを有効化（既に有効）
gradient_checkpointing=True
```

### Exercise 6: MLflow実験が見つからない ⭐ **NEW**

**原因**: 実験名のパスが正しくない

**解決策**:
```
# ユーザー名を動的に取得
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{username}/llm-evaluation"
mlflow.set_experiment(experiment_name)
```

## 📚 参考資料

### 公式ドキュメント
- [Databricks Foundation Model APIs](https://docs.databricks.com/machine-learning/foundation-model-apis/)
- [Databricks Structured Outputs](https://docs.databricks.com/machine-learning/model-serving/structured-outputs)
- [Databricks Function Calling](https://docs.databricks.com/machine-learning/model-serving/function-calling)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) ⭐ **NEW**
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) ⭐ **NEW**
- [Gemma 3 Model Card](https://huggingface.co/google/gemma-3-270m-it)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

### 技術ブログ・論文
- [Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Transformerの原論文
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Databricks: Introducing Structured Outputs](https://www.databricks.com/blog/introducing-structured-outputs-batch-and-agent-workflows)
- [MLflow: A Platform for Managing the Machine Learning Lifecycle](https://mlflow.org/docs/latest/index.html) ⭐ **NEW**

## 🤝 コントリビューション

バグ報告、改善提案、新しいExerciseのアイデアなど、コントリビューションを歓迎します。

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-exercise`)
3. 変更をコミット (`git commit -m 'Add amazing exercise'`)
4. ブランチにプッシュ (`git push origin feature/amazing-exercise`)
5. Pull Requestを作成

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 謝辞

本教材は以下のリソースを参考にしています：
- Databricks公式ドキュメントとサンプルコード
- Google Gemma 3チームによるモデルとドキュメント
- HuggingFace TransformersとPEFTライブラリ
- Databricks Dolly 15k日本語訳（kunishou氏）とgozarinne版（bbz662bbz氏）
- MLflow開発チームとコミュニティ ⭐ **NEW**

---

**Happy Learning! 🚀**

**Powered by Databricks + MLflow** ⭐
