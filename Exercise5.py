# Databricks notebook source
# MAGIC %md
# MAGIC # Exercise 5: Gemma 3 270M の LoRA ファインチューニング（Unsloth 版）
# MAGIC
# MAGIC ## 目的
# MAGIC - Unsloth を使った高速・省メモリな LoRA ファインチューニングを体験する
# MAGIC - 日本語データセット（ござる口調 Dolly）での Instruction Tuning を実践する
# MAGIC - ファインチューニング前後での応答の変化を比較する
# MAGIC - Databricks の GPU 環境上で Unsloth を動かしてみる
# MAGIC
# MAGIC ## 使用するもの
# MAGIC - **ベースモデル**: `unsloth/gemma-3-270m-it`
# MAGIC - **データセット**: `bbz662bbz/databricks-dolly-15k-ja-gozarinnemon`
# MAGIC   - Databricks Dolly 15k 日本語訳版（語尾が「ござる」）
# MAGIC - **手法**: LoRA（Parameter-Efficient Fine-Tuning）
# MAGIC - **ライブラリ**: Unsloth, TRL, Datasets, Transformers
# MAGIC
# MAGIC ## 全体の流れ
# MAGIC 1. 環境確認・ライブラリインストール
# MAGIC 2. データセットのロードとチャット形式への変換
# MAGIC 3. Unsloth で Gemma 3 270M モデル＋LoRA の準備
# MAGIC 4. ファインチューニング前の性能評価
# MAGIC 5. SFTTrainer + Unsloth でトレーニング
# MAGIC 6. ファインチューニング後の性能評価
# MAGIC 7. モデルの保存

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: 環境確認 & ライブラリインストール

# COMMAND ----------

# DBTITLE 1,ライブラリインストール（必要に応じて）
# MAGIC %pip install -U unsloth trl transformers datasets accelerate bitsandbytes
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,環境確認
import torch, platform, sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: データセットのロード
# MAGIC
# MAGIC - Databricks Dolly 15k 日本語訳版（ござる口調）を利用します。
# MAGIC - `instruction` / `input` / `output` の3つを、あとで会話形式に変換します。

# COMMAND ----------

from datasets import load_dataset

dataset_name = "bbz662bbz/databricks-dolly-15k-ja-gozarinnemon"

print("【データセットのロード】")
print(f"Dataset: {dataset_name}\n")

dataset = load_dataset(dataset_name, split="train")

print("✅ データセットロード完了")
print(f"サンプル数: {len(dataset):,}")
print("\n【データセットの構造】")
print(dataset)

# COMMAND ----------

# DBTITLE 1,サンプル表示
import random

print("【データセットのサンプル】\n")
for i in range(3):
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    print(f"サンプル {i+1}:")
    print(f"カテゴリー: {sample.get('category', 'N/A')}")
    print(f"instruction: {sample['instruction']}")
    print(f"input      : {sample.get('input', '')}")
    print(f"output     : {sample['output']}")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Dolly 形式 → Gemma3 会話形式（Unsloth スタイル）への変換
# MAGIC
# MAGIC - Dolly: `instruction` + `input` → user、`output` → assistant
# MAGIC - Unsloth の Gemma3 テンプレートに合わせて `conversations` カラムを作成します。
# MAGIC - さらに `tokenizer.apply_chat_template` で最終的な `text` カラムを作り、
# MAGIC   TRL の `SFTTrainer` に渡します。:contentReference[oaicite:1]{index=1}

# COMMAND ----------

def dolly_to_conversations(example):
    """Dolly 形式を conversations 形式に変換"""
    if example.get("input") and example["input"].strip():
        user_content = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_content = example["instruction"]

    system_prompt = (
        "あなたは親切で丁寧な日本語のアシスタントでござる。"
        "ユーザーの質問に、わかりやすく、かつ語尾を「〜でござる」「〜でござるか」などの"
        "ござる口調で回答するでござる。"
    )

    conversations = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"conversations": conversations}

print("【Dolly → conversations 変換中】")
dataset = dataset.map(
    dolly_to_conversations,
    remove_columns=dataset.column_names,  # 元のカラムは削除して conversations のみに
)
print("✅ 変換完了\n")
print("【変換後サンプル（conversations）】")
print(dataset[0]["conversations"])

# COMMAND ----------

# DBTITLE 1,Train/Test スプリット
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print("【データセット分割】")
print(f"訓練サンプル数: {len(dataset['train']):,}")
print(f"テストサンプル数: {len(dataset['test']):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Unsloth で Gemma 3 270M モデルをロード
# MAGIC
# MAGIC - `FastModel.from_pretrained` でベースモデル＋トークナイザーを取得
# MAGIC - `get_chat_template(..., chat_template="gemma3")` で Gemma3 用テンプレートをセット
# MAGIC - `FastModel.get_peft_model` で LoRA 設定を追加します。:contentReference[oaicite:2]{index=2}

# COMMAND ----------

import os
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

max_seq_length = 1024
model_name = "unsloth/gemma-3-270m-it"

print("【Unsloth モデルのロード】")
print(f"Model ID: {model_name}\n")

model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,      # 4bit 量子化（VRAM が厳しければ True 推奨）
    load_in_8bit=False,
    full_finetuning=False,
)

# Gemma3 用のチャットテンプレートを設定
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma3",
)

print("✅ モデル & トークナイザー ロード完了")

# COMMAND ----------

# DBTITLE 1,LoRA 設定（Unsloth）
model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("✅ LoRA 設定完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.5: conversations → text への変換（テンプレート適用）
# MAGIC
# MAGIC - Unsloth / Gemma3 推奨の形式で `text` カラムを作成します。
# MAGIC - `SFTTrainer` はこの `text` をもとにトークナイズします。:contentReference[oaicite:3]{index=3}

# COMMAND ----------

def apply_gemma3_template(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

print("【conversations → text 変換中】")
dataset["train"] = dataset["train"].map(apply_gemma3_template, batched=True)
dataset["test"]  = dataset["test"].map(apply_gemma3_template, batched=True)

print("✅ 変換完了")
print("【text サンプル】")
print(dataset["train"][0]["text"][:500], " ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: ファインチューニング前の性能評価
# MAGIC
# MAGIC - いくつかの質問に対する応答を確認し、後で比較します。

# COMMAND ----------

def test_model(model, tokenizer, test_prompts, title="ファインチューニング前の性能"):
    print(f"\n")
    model.eval()
    for i, prompt in enumerate(test_prompts, 1):
        messages = [
            {
                "role": "system",
                "content": "あなたは親切な日本語アシスタントでござる。"
                           "語尾を「〜でござる」にして丁寧に回答するでござる。",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).removeprefix("<bos>")

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        print(f"質問 {i}: {prompt}")
        print(f"回答: {response}")
        print("-" * 80 + "\n")

test_prompts = [
    "機械学習とは何ですか？",
    "日本の首都はどこですか？",
    "Pythonの特徴を教えてください。",
]

test_model(model, tokenizer, test_prompts, title="ファインチューニング前の性能")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: SFTTrainer 用の設定（Unsloth スタイル + MLflow ロギング）
# MAGIC
# MAGIC - `dataset_text_field="text"` で、さきほど作ったテキストを学習に使用します。
# MAGIC - `report_to="mlflow"` とすることで、Trainer から MLflow に loss などのメトリクスが自動ログされます。
# MAGIC - Databricks では、このノートブックに紐づいた MLflow Experiment に記録されます。
# MAGIC - そのほかは元のノートブックと同等のエポック数・バッチサイズにしています。

# COMMAND ----------

from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

output_dir = "/tmp/gemma-3-270m-unsloth-gozaru"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),
    optim="paged_adamw_8bit",
    report_to="mlflow",         # ★ これでデータブリックス上の "mlflow" に記録されます
)

sft_config = SFTConfig(
    **training_args.to_dict(),
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
)

print("【トレーニング設定】")
print(f"エポック数: {sft_config.num_train_epochs}")
print(f"バッチサイズ: {sft_config.per_device_train_batch_size}")
print(f"勾配累積ステップ: {sft_config.gradient_accumulation_steps}")
print(f"実効バッチサイズ: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"学習率: {sft_config.learning_rate}")
print(f"出力ディレクトリ: {output_dir}")
print(f"MLflow report_to: {sft_config.report_to}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Unsloth の `train_on_responses_only` でラベルマスキング
# MAGIC
# MAGIC - `user` 側のトークンには loss をかけず、`assistant` 側のみを学習します。
# MAGIC - Instruction に対する回答品質が向上しやすくなります。:contentReference[oaicite:5]{index=5}

# COMMAND ----------

from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=sft_config,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

print("✅ SFTTrainer 準備完了")
print("【トレーニング開始】")

# COMMAND ----------

trainer_stats = trainer.train()

print("✅ トレーニング完了")
print(trainer_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6.5: MLflow へのロギング確認
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
# MAGIC ## Step 7: ファインチューニング後の性能評価
# MAGIC
# MAGIC - Step 4 と同じプロンプトで再度テストし、違いを確認します。

# COMMAND ----------

test_model(model, tokenizer, test_prompts, title="ファインチューニング後の性能")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: モデルの保存
# MAGIC
# MAGIC - LoRA アダプタとして保存するか、16bit / 4bit にマージして保存するかを選びます。:contentReference[oaicite:6]{index=6}

# COMMAND ----------

# DBTITLE 1,LoRA アダプタとして保存
model.save_pretrained("gemma-3-270m-unsloth-gozaru")
tokenizer.save_pretrained("gemma-3-270m-unsloth-gozaru")
print("✅ LoRA アダプタを保存しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: モデルのファインチューニング済みモデルのロードと実行
# MAGIC
# MAGIC - LoRA アダプタを読み込んで、モデルにマージして、モデルを推論する。

# COMMAND ----------

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

model_path = "gemma-3-270m-unsloth-gozaru"

# LoRAアダプターとトークナイザーのロード
model, tokenizer = FastModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

# 推論例
prompt = "日本の首都はどこ？"
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
).removeprefix("<bos>")

inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )
generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"質問: {prompt}")
print(f"回答: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎓 総括（Unsloth 版）
# MAGIC
# MAGIC - Unsloth を用いることで、少ないコード量で
# MAGIC   - 4bit 量子化
# MAGIC   - LoRA の適用
# MAGIC   - `train_on_responses_only` による効率的な損失計算
# MAGIC   をまとめて扱えるようになりました。
# MAGIC - Databricks 上でも、Unsloth の公式ノートブックとほぼ同じ構成で Gemma 3 270M の
# MAGIC   日本語 Instruction Tuning（ござる口調）を体験できます。
# MAGIC - 次回の AI エージェント講義では、この Unsloth 版ファインチューニング済みモデルを
# MAGIC   **エージェントのバックエンドモデル**として活用していきましょう。
