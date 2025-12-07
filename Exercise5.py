# Databricks notebook source
# MAGIC %md
# MAGIC # Exercise 5: Gemma 3 270MのLoRAファインチューニング
# MAGIC
# MAGIC ## 目的
# MAGIC - LoRA（Low-Rank Adaptation）を使った効率的なファインチューニングを学ぶ
# MAGIC - 日本語データセットでのInstruction Tuningを実践する
# MAGIC - ファインチューニング前後のモデル性能を比較する
# MAGIC - Databricks環境でのGPUトレーニングを体験する
# MAGIC
# MAGIC ## 使用するもの
# MAGIC - **ベースモデル**: unsloth/gemma-3-270m-it
# MAGIC - **データセット**: bbz662bbz/databricks-dolly-15k-ja-gozarinnemon
# MAGIC   - Databricks Dolly 15kの日本語訳版で、回答の語尾が「ござる」口調
# MAGIC - **手法**: LoRA（Parameter-Efficient Fine-Tuning）
# MAGIC - **ライブラリ**: Hugging Face Transformers, PEFT, TRL

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚠️ 重要: GPU環境の確認
# MAGIC
# MAGIC このNotebookは**GPU必須**です。以下のいずれかを使用してください：
# MAGIC
# MAGIC 1. **Serverless GPU**（推奨）
# MAGIC    - NotebookのConnect → Serverless GPU
# MAGIC    - A10またはH100を選択
# MAGIC
# MAGIC 2. **Single Node GPU Cluster**
# MAGIC    - g4dn.xlarge以上のインスタンス
# MAGIC    - Databricks Runtime 14.3 ML以降

# COMMAND ----------

# DBTITLE 1,GPU確認
import torch

print("【GPU環境確認】")
if torch.cuda.is_available():
    print(f"✅ GPU利用可能")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"総メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA バージョン: {torch.version.cuda}")
else:
    print("❌ GPUが利用できません")
    print("このNotebookを実行するにはGPU環境が必要です")
    raise RuntimeError("GPU not available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install --upgrade transformers datasets accelerate peft trl bitsandbytes sentencepiece hf_transfer
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

# H100 を 1枚だけ見せる（ここでは GPU0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ついでに OOM 時の断片化対策も入れておく
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# COMMAND ----------

# DBTITLE 1,インストール確認
import transformers
import datasets
import peft
import trl
import torch

print("【インストールされたバージョン】")
print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")
print(f"peft: {peft.__version__}")
print(f"trl: {trl.__version__}")
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: データセットの準備

# COMMAND ----------

# DBTITLE 1,データセットのロード
from datasets import load_dataset

dataset_name = "bbz662bbz/databricks-dolly-15k-ja-gozarinnemon"

print(f"【データセットのロード】")
print(f"Dataset: {dataset_name}\n")

# データセットをロード
dataset = load_dataset(dataset_name, split="train")

print(f"✅ データセットロード完了")
print(f"サンプル数: {len(dataset):,}")
print(f"\n【データセットの構造】")
print(dataset)

# COMMAND ----------

# DBTITLE 1,データセットのサンプル確認
import random

# ランダムに3つのサンプルを表示
print("【データセットのサンプル】\n")

for i in range(3):
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    
    print(f"サンプル {i+1}:")
    print(f"カテゴリー: {sample.get('category', 'N/A')}")
    print(f"指示: {sample['instruction']}")
    
    if sample.get('input'):
        print(f"入力: {sample['input']}")
    
    print(f"出力: {sample['output']}")
    print("-" * 80 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: データセットをチャット形式に変換

# COMMAND ----------

# DBTITLE 1,チャット形式への変換関数
def format_to_chat(example):
    """
    Dolly形式のデータをチャット形式に変換
    
    Dolly形式:
    - instruction: 指示
    - input: コンテキスト（オプション）
    - output: 期待される出力
    
    チャット形式:
    - messages: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """
    # ユーザーメッセージの構築
    if example.get('input') and example['input'].strip():
        # inputがある場合は、instruction + input
        user_content = f"{example['instruction']}\n\n{example['input']}"
    else:
        # inputがない場合は、instructionのみ
        user_content = example['instruction']
    
    # チャット形式に変換
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['output']}
    ]
    
    return {"messages": messages}

# データセット全体を変換
print("【データセットをチャット形式に変換中】")
dataset = dataset.map(
    format_to_chat,
    remove_columns=dataset.column_names
)

print("✅ 変換完了\n")
print("【変換後のサンプル】")
print(dataset[0]['messages'])

# COMMAND ----------

# DBTITLE 1,Train/Testスプリット
# 80%を訓練用、20%をテスト用に分割
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print("【データセット分割】")
print(f"訓練サンプル数: {len(dataset['train']):,}")
print(f"テストサンプル数: {len(dataset['test']):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: ベースモデルとトークナイザーのロード

# COMMAND ----------

# DBTITLE 1,モデルとトークナイザーのロード
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

import os

model_id = "unsloth/gemma-3-270m-it"

print(f"【モデルのロード】")
print(f"Model ID: {model_id}\n")

# トークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(model_id)

# パディングトークンの設定（LoRAトレーニングに必要）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit量子化設定（メモリ削減）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# モデルのロード（4-bit量子化）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    # device_map="auto",
    dtype=torch.bfloat16,
)
model.to("cuda")  # ← ここ足してもいいです

print(f"✅ モデルロード完了")
print(f"パラメータ数: {model.num_parameters():,}")

# メモリ使用量確認
allocated = torch.cuda.memory_allocated(0) / 1024**3
print(f"GPUメモリ使用量: {allocated:.2f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 4-bit量子化の効果
# MAGIC
# MAGIC - **通常（FP16）**: 約540MB
# MAGIC - **4-bit量子化**: 約135MB
# MAGIC - **メモリ削減率**: 約75%
# MAGIC
# MAGIC これにより、小規模GPUでも大規模モデルのファインチューニングが可能になります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: LoRA設定

# COMMAND ----------

# DBTITLE 1,LoRA設定の定義
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training

# モデルを量子化トレーニング用に準備
model = prepare_model_for_kbit_training(model)

# LoRA設定
lora_config = LoraConfig(
    r=16,                                    # LoRAのランク（低ランク行列の次元）
    lora_alpha=32,                          # スケーリングファクター
    target_modules=["q_proj", "v_proj"],   # LoRAを適用するモジュール
    lora_dropout=0.05,                      # ドロップアウト率
    bias="none",                            # バイアスの扱い
    task_type=TaskType.CAUSAL_LM           # タスクタイプ
)

print("【LoRA設定】")
print(f"ランク (r): {lora_config.r}")
print(f"Alpha: {lora_config.lora_alpha}")
print(f"対象モジュール: {lora_config.target_modules}")
print(f"ドロップアウト: {lora_config.lora_dropout}")
print(f"タスクタイプ: {lora_config.task_type}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 LoRAパラメータの説明
# MAGIC
# MAGIC | パラメータ | 説明 | 推奨値 |
# MAGIC |----------|------|--------|
# MAGIC | **r** | LoRAランク（低ランク行列の次元） | 8-32（小さいほど効率的、大きいほど表現力が高い） |
# MAGIC | **lora_alpha** | スケーリングファクター | 通常は`r`の2倍 |
# MAGIC | **target_modules** | LoRAを適用する層 | Attention層（q_proj, v_proj）が一般的 |
# MAGIC | **lora_dropout** | 過学習防止 | 0.05-0.1 |
# MAGIC
# MAGIC **訓練可能パラメータの削減**:
# MAGIC - 全パラメータファインチューニング: 270M（100%）
# MAGIC - LoRA（r=16）: 約0.5-1M（0.2-0.4%）

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: ファインチューニング前の性能評価

# COMMAND ----------

# DBTITLE 1,ファインチューニング前のテスト
def test_model(model, tokenizer, test_prompts):
    """モデルをテストする関数"""
    print("【ファインチューニング前の性能】\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"質問 {i}: {prompt}")
        print(f"回答: {response}")
        print("-" * 80 + "\n")

# COMMAND ----------

# テストプロンプト
test_prompts = [
    "機械学習とは何ですか？",
    "日本の首都はどこですか？",
    "Pythonの特徴を教えてください。"
]

test_model(model, tokenizer, test_prompts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: トレーニング設定

# COMMAND ----------

# DBTITLE 1,トレーニング引数の設定
from transformers import TrainingArguments

# 出力ディレクトリの設定
output_dir = "/tmp/gemma-3-270m-finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,                    # エポック数
    per_device_train_batch_size=4,        # バッチサイズ
    gradient_accumulation_steps=1,         # 勾配累積ステップ（実効バッチサイズ=16）
    learning_rate=2e-4,                    # 学習率
    lr_scheduler_type="cosine",            # 学習率スケジューラー
    warmup_ratio=0.1,                      # ウォームアップ比率
    logging_steps=10,                      # ログ出力頻度
    save_strategy="epoch",                 # 保存戦略
    eval_strategy="epoch",                 # 評価戦略
    bf16=True,                             # BFloat16精度
    fp16=False,                            # FP16精度
    gradient_checkpointing=True,           # 勾配チェックポイント（メモリ削減）
    remove_unused_columns=False,           # 未使用列を削除しない
    report_to="mlflow",                      # レポート先（MLflow）
    
    # 追加する重要な設定
    # dataloader_num_workers=4,        # データローディングの並列化
    # dataloader_pin_memory=True,       # GPU転送の高速化
    # ddp_find_unused_parameters=False, # DDP最適化
)

print("【トレーニング設定】")
print(f"エポック数: {training_args.num_train_epochs}")
print(f"バッチサイズ: {training_args.per_device_train_batch_size}")
print(f"勾配累積ステップ: {training_args.gradient_accumulation_steps}")
print(f"実効バッチサイズ: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"学習率: {training_args.learning_rate}")
print(f"出力ディレクトリ: {output_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 トレーニングパラメータの調整
# MAGIC
# MAGIC **メモリ不足エラーが発生した場合**:
# MAGIC 1. `per_device_train_batch_size`を2に減らす
# MAGIC 2. `gradient_accumulation_steps`を8に増やす（実効バッチサイズ維持）
# MAGIC 3. `gradient_checkpointing=True`を有効にする（既に有効）
# MAGIC
# MAGIC **トレーニング時間を短縮したい場合**:
# MAGIC 1. `num_train_epochs`を1-2に減らす
# MAGIC 2. データセットをサブサンプリング（次のセルで実装）

# COMMAND ----------

# DBTITLE 1,（オプション）データセットのサブサンプリング
# 高速テスト用に、データセットの一部のみを使用する場合
USE_SUBSET = False  # Trueにすると10%のみ使用

if USE_SUBSET:
    dataset['train'] = dataset['train'].select(range(int(len(dataset['train']) * 0.1)))
    dataset['test'] = dataset['test'].select(range(int(len(dataset['test']) * 0.1)))
    print(f"【サブセット使用】")
    print(f"訓練サンプル数: {len(dataset['train']):,}")
    print(f"テストサンプル数: {len(dataset['test']):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: SFTTrainerによるファインチューニング

# COMMAND ----------

# DBTITLE 1,SFTTrainerの作成と実行
from trl import SFTTrainer, SFTConfig

# SFT（Supervised Fine-Tuning）設定
sft_config = SFTConfig(
    **training_args.to_dict(),
    max_length=1024,                   # 最大シーケンス長
    packing=False,                         # パッキング無効
)

# SFTTrainerの作成
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("✅ SFTTrainer作成完了")
print("\n【トレーニング開始】")
print("このプロセスには10-30分かかります...\n")

# トレーニング実行
trainer.train()

print("\n✅ トレーニング完了！")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 トレーニングメトリクスの確認
# MAGIC
# MAGIC トレーニング中、以下のメトリクスが出力されます：
# MAGIC - **loss**: 訓練損失（低いほど良い）
# MAGIC - **learning_rate**: 現在の学習率
# MAGIC - **epoch**: 現在のエポック
# MAGIC
# MAGIC 正常なトレーニングでは、lossが徐々に減少します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7.5: MLflow へのロギング確認
# MAGIC
# MAGIC - 直近の MLflow run ID を取得して表示します。
# MAGIC - Databricks の「Experiments」タブからこの run を開くと、`loss` の推移グラフを確認できます。

# COMMAND ----------

import mlflow

last_run = mlflow.last_active_run()
if last_run is not None:
    print("✅ MLflow ログ完了")
    print(f"Run ID   : {last_run.info.run_id}")
    print(f"Run Name : {last_run.info.run_name}")
    print(f"Experiment ID : {last_run.info.experiment_id}")
else:
    print("⚠️ MLflow のアクティブな run が見つかりませんでした。")
    print("- `TrainingArguments(report_to='mlflow')` が設定されているか")
    print("- ランタイムを再起動した場合は Step 5〜6 を再実行したか")
    print("を確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: ファインチューニング後の性能評価

# COMMAND ----------

# DBTITLE 1,ファインチューニング後のテスト
print("【ファインチューニング後の性能】\n")

for i, prompt in enumerate(test_prompts, 1):
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"質問 {i}: {prompt}")
    print(f"回答: {response}")
    print("-" * 80 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 期待される結果
# MAGIC
# MAGIC ファインチューニング後、モデルは以下の特徴を示すはずです：
# MAGIC - **語尾が「ござる」口調**になる
# MAGIC - より詳細で構造化された回答
# MAGIC - データセットのスタイルに適応した表現

# COMMAND ----------

# DBTITLE 1,データセット由来のプロンプトでテスト
# データセットからランダムにサンプルを選択してテスト
import random

print("【データセット由来のプロンプトでテスト】\n")

for _ in range(3):
    idx = random.randint(0, len(dataset['test']) - 1)
    sample = dataset['test'][idx]['messages']
    
    user_prompt = sample[0]['content']
    expected_output = sample[1]['content']
    
    messages = [{"role": "user", "content": user_prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"質問: {user_prompt[:100]}...")
    print(f"\n期待される回答: {expected_output[:150]}...")
    print(f"\nモデルの回答: {response}")
    print("=" * 80 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: モデルの保存

# COMMAND ----------

# DBTITLE 1,ファインチューニング済みモデルの保存
# LoRAアダプターのみを保存（元のモデル重みは保存不要）
adapter_save_dir = "gemma-3-270m-lora-adapters"

trainer.model.save_pretrained(adapter_save_dir)
tokenizer.save_pretrained(adapter_save_dir)

print(f"✅ LoRAアダプターを保存しました")
print(f"保存先: {adapter_save_dir}")
print(f"\n保存されたファイル:")

import os
for file in os.listdir(adapter_save_dir):
    file_path = os.path.join(adapter_save_dir, file)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / 1024**2
        print(f"  {file}: {size:.2f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 LoRA保存の利点
# MAGIC
# MAGIC - **元のモデル（270M）**: 約540MB
# MAGIC - **LoRAアダプターのみ**: 約5-10MB
# MAGIC - **削減率**: 約98%
# MAGIC
# MAGIC 推論時は、ベースモデル + LoRAアダプターをロードするだけで使用可能

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: 保存したモデルのロード（推論用）

# COMMAND ----------

# DBTITLE 1,保存したLoRAモデルのロード
from peft import AutoPeftModelForCausalLM

# 新しいセッションでモデルをロードする場合
print("【保存したLoRAモデルのロード】")

# LoRAアダプターを含むモデルをロード
inference_model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_save_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

inference_tokenizer = AutoTokenizer.from_pretrained(adapter_save_dir)

print("✅ モデルロード完了")

# テスト
test_message = [{"role": "user", "content": "Pythonの特徴を教えてください。"}]

inputs = inference_tokenizer.apply_chat_template(
    test_message,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(inference_model.device)

with torch.no_grad():
    outputs = inference_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=inference_tokenizer.eos_token_id
    )

generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
response = inference_tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n【ロードしたモデルでのテスト】")
print(f"質問: {test_message[0]['content']}")
print(f"回答: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Exercise 5のまとめ
# MAGIC
# MAGIC このExerciseで学んだこと：
# MAGIC
# MAGIC ### ファインチューニング技術
# MAGIC 1. **LoRA（Low-Rank Adaptation）**
# MAGIC    - パラメータ効率的なファインチューニング
# MAGIC    - 訓練パラメータを99%以上削減
# MAGIC    - 元のモデル重みは凍結
# MAGIC
# MAGIC 2. **4-bit量子化**
# MAGIC    - メモリ使用量を75%削減
# MAGIC    - 小規模GPUでのトレーニングを可能に
# MAGIC    - BitsAndBytesConfigによる設定
# MAGIC
# MAGIC 3. **Supervised Fine-Tuning (SFT)**
# MAGIC    - TRLのSFTTrainerによる簡潔な実装
# MAGIC    - チャット形式データでのInstruction Tuning
# MAGIC    - 評価とチェックポイント保存の自動化
# MAGIC
# MAGIC ### データセット処理
# MAGIC 4. **チャット形式への変換**
# MAGIC    - Dolly形式からOpenAI形式へ
# MAGIC    - apply_chat_templateとの統合
# MAGIC
# MAGIC ### モデル管理
# MAGIC 5. **効率的な保存とロード**
# MAGIC    - LoRAアダプターのみを保存（約5-10MB）
# MAGIC    - AutoPeftModelForCausalLMによるロード
# MAGIC    - 本番環境への容易なデプロイ

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 ファインチューニング前後の比較
# MAGIC
# MAGIC | 項目 | 前 | 後 |
# MAGIC |------|-----|-----|
# MAGIC | **語尾** | 通常の日本語 | 「ござる」口調 |
# MAGIC | **回答スタイル** | 汎用的 | データセットに適応 |
# MAGIC | **詳細度** | 簡潔 | より詳細 |
# MAGIC | **パラメータ** | 270M | 270M + 0.5M（LoRA） |
# MAGIC | **メモリ** | 540MB | 545MB |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💡 次のステップ
# MAGIC
# MAGIC ファインチューニングをマスターしたら、以下にチャレンジ：
# MAGIC
# MAGIC 1. **独自データセットの作成**
# MAGIC    - 自社のQ&Aデータでファインチューニング
# MAGIC    - 特定ドメイン（医療、法律、技術）への特化
# MAGIC
# MAGIC 2. **ハイパーパラメータチューニング**
# MAGIC    - LoRAランク（r）の最適化
# MAGIC    - 学習率とエポック数の調整
# MAGIC    - target_modulesの拡張
# MAGIC
# MAGIC 3. **評価指標の実装**
# MAGIC    - ROUGE、BLEU、BERTScoreでの自動評価
# MAGIC    - 人間評価との相関分析
# MAGIC
# MAGIC 4. **Databricks Model Servingへのデプロイ**
# MAGIC    - ファインチューニング済みモデルのAPIエンドポイント化
# MAGIC    - A/Bテストによる性能比較
# MAGIC
# MAGIC 5. **マルチタスク学習**
# MAGIC    - 複数のデータセットを混合してトレーニング
# MAGIC    - タスク間の転移学習効果の検証

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎓 総括
# MAGIC
# MAGIC Exercise 5では、**LoRAを使ったパラメータ効率的なファインチューニング**を学びました。
# MAGIC
# MAGIC これにより：
# MAGIC - 少ないリソースで大規模モデルをカスタマイズ可能
# MAGIC - 独自データでのスタイル・ドメイン適応を実現
# MAGIC - 実務で使える効率的なモデル管理手法を習得
# MAGIC
# MAGIC 次回のAIエージェント講義では、このファインチューニング済みモデルを
# MAGIC **エージェントシステムのコンポーネント**として活用します！
