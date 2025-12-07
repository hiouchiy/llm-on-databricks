# Databricks notebook source
# MAGIC %md
# MAGIC # Exercise 4: HuggingFaceモデルのローカル実行（Gemma 3 270M）
# MAGIC
# MAGIC ## 目的
# MAGIC - HuggingFace Hubから直接モデルをダウンロードして使用する方法を学ぶ
# MAGIC - `apply_chat_template`を使った正しいチャット形式の実装
# MAGIC - Foundation Model API以外の選択肢（オープンソースモデルのローカル実行）を理解する
# MAGIC - 軽量モデル（270M）の実用性を評価する
# MAGIC
# MAGIC ## Gemma 3 270Mについて
# MAGIC - **パラメータ数**: 270M（2.7億）
# MAGIC - **特徴**: Gemma 3ファミリーの最小モデル、モバイル・エッジデバイス向け
# MAGIC - **用途**: 質問応答、要約、分類、軽量推論タスク
# MAGIC - **メモリ**: FP16で約540MB、INT8で約270MB

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境確認とセットアップ

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install --upgrade transformers accelerate sentencepiece
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,GPU確認
import torch

print("【GPUデバイス情報】")
if torch.cuda.is_available():
    print(f"✅ GPU利用可能")
    print(f"GPU数: {torch.cuda.device_count()}")
    print(f"現在のGPU: {torch.cuda.current_device()}")
    print(f"GPU名: {torch.cuda.get_device_name(0)}")
    print(f"CUDA バージョン: {torch.version.cuda}")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"総メモリ: {total_memory:.2f} GB")
else:
    print("⚠️ GPUが利用できません。CPUで実行されます。")

# COMMAND ----------

# DBTITLE 1,インストール確認
import transformers
import torch

print("【インストールされたバージョン】")
print(f"transformers: {transformers.__version__}")
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: モデルとトークナイザーのロード

# COMMAND ----------

# DBTITLE 1,モデルとトークナイザーのロード
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os

model_id = "unsloth/gemma-3-270m-it" #元ネタは"google/gemma-3-270m-it"

print(f"【モデルのダウンロード中】")
print(f"Model ID: {model_id}")
print("初回実行時は1-2分かかります...\n")

# トークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(model_id)

# モデルのロード
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print(f"✅ モデルのロード完了")
print(f"デバイス: {device}")
print(f"データ型: {model.dtype}")
print(f"パラメータ数: {model.num_parameters():,}")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"GPUメモリ使用量: {allocated:.2f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: apply_chat_templateを使った正しい使い方
# MAGIC
# MAGIC Gemma 3はチャット用のテンプレートを内蔵しており、
# MAGIC `apply_chat_template`を使うことで適切なフォーマットが自動適用されます。

# COMMAND ----------

# DBTITLE 1,基本的な会話（低レベルAPI）
def chat_with_model(messages: list, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
    """
    apply_chat_templateを使った会話生成
    
    Args:
        messages: OpenAI形式のメッセージリスト
        max_new_tokens: 生成する最大トークン数
        temperature: 生成のランダム性
    
    Returns:
        モデルの応答テキスト
    """
    # チャットテンプレートを適用してトークン化
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # モデルの応答プロンプトを追加
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 入力部分を除いて、生成されたテキストのみをデコード
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response

# テスト実行
messages = [
    {"role": "user", "content": "日本の最も北にある都道府県は？"}
]

print("【ユーザー】")
print(messages[0]["content"])
print("\n【モデル】")
response = chat_with_model(messages, max_new_tokens=150, temperature=0.7)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 apply_chat_templateのポイント
# MAGIC
# MAGIC - `add_generation_prompt=True`: モデルの応答を促すプロンプトを自動追加
# MAGIC - `tokenize=True, return_tensors="pt"`: 直接PyTorchテンソルで返す
# MAGIC - `return_dict=True`: 辞書形式で返す（`model.generate()`に直接渡せる）
# MAGIC - 出力から入力部分を除外: `outputs[0][inputs["input_ids"].shape[-1]:]`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Pipelineを使ったシンプルな方法

# COMMAND ----------

# DBTITLE 1,Pipeline APIによる会話
from transformers import pipeline

# チャット用のpipelineを作成
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

print("【Pipeline APIを使用した会話】\n")

# メッセージ形式で会話
messages = [
    {"role": "user", "content": "機械学習とは何ですか？簡潔に説明してください。"}
]

result = pipe(messages)

print("【ユーザー】")
print(messages[0]["content"])
print("\n【モデル】")
# Pipelineは生成されたテキスト全体を返すので、最後のアシスタント応答を抽出
print(result[0]["generated_text"][-1]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: マルチターン会話
# MAGIC
# MAGIC 会話履歴を保持した対話を実装します。

# COMMAND ----------

# DBTITLE 1,マルチターン会話の実装
def multi_turn_chat(conversation_history: list, user_message: str) -> tuple:
    """
    マルチターン会話
    
    Args:
        conversation_history: これまでの会話履歴
        user_message: 新しいユーザーメッセージ
    
    Returns:
        (モデルの応答, 更新された会話履歴)
    """
    # ユーザーメッセージを追加
    conversation_history.append({"role": "user", "content": user_message})
    
    # モデルの応答を生成
    response = chat_with_model(conversation_history, max_new_tokens=150)
    
    # アシスタントの応答を履歴に追加
    conversation_history.append({"role": "assistant", "content": response})
    
    return response, conversation_history

# 会話の開始
conversation = []

print("=" * 60)
print("マルチターン会話のデモ")
print("=" * 60)

# ターン1
print("\n【ターン1】")
user_msg_1 = "決定木アルゴリズムの利点を教えてください"
print(f"ユーザー: {user_msg_1}")
assistant_msg_1, conversation = multi_turn_chat(conversation, user_msg_1)
print(f"モデル: {assistant_msg_1}")

# ターン2（文脈を参照）
print("\n【ターン2】")
user_msg_2 = "それでは、欠点は何ですか？"
print(f"ユーザー: {user_msg_2}")
assistant_msg_2, conversation = multi_turn_chat(conversation, user_msg_2)
print(f"モデル: {assistant_msg_2}")

# ターン3
print("\n【ターン3】")
user_msg_3 = "その欠点を克服する方法はありますか？"
print(f"ユーザー: {user_msg_3}")
assistant_msg_3, conversation = multi_turn_chat(conversation, user_msg_3)
print(f"モデル: {assistant_msg_3}")

print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 文脈の保持を確認
# MAGIC
# MAGIC - ターン2の「それ」が「決定木」を指していることを理解
# MAGIC - ターン3の「その欠点」が前の応答を参照
# MAGIC - これがapply_chat_templateによる正しい会話管理

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: 様々なタスクでの評価

# COMMAND ----------

# DBTITLE 1,タスク1: 質問応答
print("=" * 60)
print("タスク1: 質問応答")
print("=" * 60 + "\n")

qa_messages = [
    {"role": "user", "content": "トランスフォーマーアーキテクチャの主要な3つの構成要素を教えてください。"}
]

qa_response = chat_with_model(qa_messages, max_new_tokens=200, temperature=0.5)

print("【質問】")
print(qa_messages[0]["content"])
print("\n【回答】")
print(qa_response)

# COMMAND ----------

# DBTITLE 1,タスク2: テキスト要約
print("=" * 60)
print("タスク2: テキスト要約")
print("=" * 60 + "\n")

long_text = """
大規模言語モデル（LLM）は、膨大なテキストデータで訓練された深層学習モデルです。
これらのモデルは、トランスフォーマーと呼ばれるニューラルネットワークアーキテクチャに基づいており、
Attentionメカニズムを使用して文脈を理解します。
GPT、BERT、LLaMA、Gemmaなどが代表的なLLMです。
近年では、数千億パラメータを持つモデルも登場し、
質問応答、文章生成、翻訳、コード生成など、幅広いタスクで人間レベルの性能を発揮しています。
"""

summary_messages = [
    {"role": "user", "content": f"以下のテキストを1-2文で要約してください。\n\n{long_text}"}
]

summary = chat_with_model(summary_messages, max_new_tokens=100, temperature=0.3)

print("【元のテキスト】")
print(long_text)
print("\n【要約】")
print(summary)

# COMMAND ----------

# DBTITLE 1,タスク3: 感情分類
print("=" * 60)
print("タスク3: 感情分析")
print("=" * 60 + "\n")

reviews = [
    "この製品は素晴らしい！期待以上の性能です。",
    "使いにくくて最悪でした。すぐに壊れました。",
    "まあまあです。特に良くも悪くもない。"
]

for i, review in enumerate(reviews, 1):
    messages = [
        {
            "role": "user",
            "content": f"以下のレビューの感情を「ポジティブ」「ネガティブ」「中立」のいずれかで分類してください。単語のみで回答してください。\n\nレビュー: {review}"
        }
    ]
    
    sentiment = chat_with_model(messages, max_new_tokens=10, temperature=0.1)
    
    print(f"レビュー{i}: {review}")
    print(f"感情: {sentiment}\n")

# COMMAND ----------

# DBTITLE 1,タスク4: システムプロンプトの使用
print("=" * 60)
print("タスク4: システムプロンプトによる振る舞い制御")
print("=" * 60 + "\n")

# システムプロンプト付きのメッセージ
messages_with_system = [
    {
        "role": "system",
        "content": "あなたは親切な先生です。専門用語を避け、初心者にもわかりやすく説明してください。"
    },
    {
        "role": "user",
        "content": "過学習について説明してください。"
    }
]

response_with_system = chat_with_model(messages_with_system, max_new_tokens=200, temperature=0.7)

print("【システムプロンプト付き】")
print(response_with_system)

print("\n" + "-"*60 + "\n")

# システムプロンプトなし
messages_without_system = [
    {
        "role": "user",
        "content": "過学習について説明してください。"
    }
]

response_without_system = chat_with_model(messages_without_system, max_new_tokens=200, temperature=0.7)

print("【システムプロンプトなし】")
print(response_without_system)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: パフォーマンス測定

# COMMAND ----------

# DBTITLE 1,推論速度の測定
import time

def measure_inference_speed(messages: list, num_runs: int = 5) -> dict:
    """推論速度を測定"""
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        _ = chat_with_model(messages, max_new_tokens=50, temperature=0.7)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "平均時間": sum(times) / len(times),
        "最小時間": min(times),
        "最大時間": max(times)
    }

test_messages = [
    {"role": "user", "content": "人工知能の未来について教えてください。"}
]

print("【推論速度測定】")
print("5回の実行で測定中...\n")
stats = measure_inference_speed(test_messages, num_runs=5)

for key, value in stats.items():
    print(f"{key}: {value:.4f}秒")

avg_tokens_per_sec = 50 / stats["平均時間"]
print(f"\n推定スループット: {avg_tokens_per_sec:.2f} tokens/秒")

# COMMAND ----------

# DBTITLE 1,メモリ使用量の確認
if torch.cuda.is_available():
    print("【GPUメモリ使用量】")
    
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
    
    print(f"現在の割り当て: {allocated:.2f} GB")
    print(f"予約済み: {reserved:.2f} GB")
    print(f"最大割り当て: {max_allocated:.2f} GB")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    utilization = (allocated / total_memory) * 100
    print(f"メモリ利用率: {utilization:.2f}%")
else:
    print("⚠️ CPUモードで実行中")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: バッチ推論

# COMMAND ----------

# DBTITLE 1,バッチ推論の実装
def batch_chat(messages_list: list, max_new_tokens: int = 100) -> list:
    """
    複数の会話をバッチ処理
    
    Args:
        messages_list: メッセージリストのリスト
        max_new_tokens: 生成する最大トークン数
    
    Returns:
        応答のリスト
    """
    # 各メッセージをテンプレート適用
    batch_inputs = []
    for messages in messages_list:
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # まず文字列として取得
        )
        batch_inputs.append(formatted)
    
    # バッチトークン化
    inputs = tokenizer(
        batch_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    # バッチ生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 各出力から入力部分を除いてデコード
    responses = []
    for i, output in enumerate(outputs):
        input_length = inputs["input_ids"][i].shape[0]
        generated_ids = output[input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response)
    
    return responses

# バッチテスト
batch_messages = [
    [{"role": "user", "content": "機械学習を一言で表現してください。"}],
    [{"role": "user", "content": "Pythonの特徴を1つ挙げてください。"}],
    [{"role": "user", "content": "データサイエンスに必要なスキルは？"}],
]

print("【バッチ推論】")
print(f"バッチサイズ: {len(batch_messages)}\n")

start_time = time.time()
batch_responses = batch_chat(batch_messages, max_new_tokens=80)
batch_time = time.time() - start_time

for i, (messages, response) in enumerate(zip(batch_messages, batch_responses), 1):
    print(f"【プロンプト{i}】 {messages[0]['content']}")
    print(f"【応答】 {response}")
    print("-" * 60)

print(f"\nバッチ処理時間: {batch_time:.4f}秒")
print(f"1プロンプトあたり: {batch_time/len(batch_messages):.4f}秒")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Exercise 4のまとめ
# MAGIC
# MAGIC このExerciseで学んだこと：
# MAGIC
# MAGIC ### 正しいGemma 3の使用方法
# MAGIC 1. **apply_chat_templateの使用**
# MAGIC    - モデル固有のチャット形式を自動適用
# MAGIC    - `add_generation_prompt=True`で応答プロンプトを追加
# MAGIC    - OpenAI形式のメッセージで統一的に扱える
# MAGIC
# MAGIC 2. **2つのAPI**
# MAGIC    - 低レベル: `tokenizer.apply_chat_template()` + `model.generate()`
# MAGIC    - 高レベル: `pipeline("text-generation")`
# MAGIC
# MAGIC 3. **マルチターン会話**
# MAGIC    - 会話履歴をメッセージリストで管理
# MAGIC    - 文脈を保持した自然な対話
# MAGIC
# MAGIC 4. **パフォーマンス管理**
# MAGIC    - GPU/CPUの自動選択
# MAGIC    - バッチ推論による効率化
# MAGIC    - メモリ使用量のモニタリング

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Foundation Model API vs ローカルモデル
# MAGIC
# MAGIC | 項目 | Gemma 3 270M (ローカル) | Foundation Model API |
# MAGIC |------|------------------------|----------------------|
# MAGIC | **実装の簡単さ** | apply_chat_template必要 | OpenAI互換で統一 |
# MAGIC | **コスト** | GPU時間のみ（固定） | トークン課金（変動） |
# MAGIC | **レイテンシー** | 低い（ネットワーク不要） | やや高い（API呼び出し） |
# MAGIC | **モデルサイズ** | 270M（軽量） | 70B〜（高性能） |
# MAGIC | **カスタマイズ** | ファインチューニング可能 | プロンプトのみ |
# MAGIC | **メンテナンス** | 自己管理 | マネージド |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💡 使い分けのガイドライン
# MAGIC
# MAGIC ### ローカルモデル（Gemma 3など）を使うべき場合
# MAGIC - リアルタイム推論が必要（レイテンシー < 100ms）
# MAGIC - 大量の継続的な推論（コスト予測可能性）
# MAGIC - ファインチューニングが必要
# MAGIC - エッジデバイスへのデプロイ
# MAGIC - 完全なデータプライバシーが必要
# MAGIC
# MAGIC ### Foundation Model APIを使うべき場合
# MAGIC - 最高品質の出力が必要（70B+モデル）
# MAGIC - プロトタイプの迅速な開発
# MAGIC - 不定期なバッチ処理
# MAGIC - インフラ管理を避けたい
# MAGIC - 統一されたOpenAI互換APIが必要

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎓 総括
# MAGIC
# MAGIC Exercise 4では、**apply_chat_templateを使った正しいチャット形式**で
# MAGIC オープンソースモデルを扱う方法を学びました。
# MAGIC
# MAGIC これにより：
# MAGIC - モデル固有のフォーマットを意識せずに会話を実装
# MAGIC - OpenAI形式のメッセージで統一的に扱える
# MAGIC - マルチターン会話や文脈保持が容易
# MAGIC - Foundation Model APIとローカルモデルの使い分けを理解
# MAGIC
# MAGIC 次回のAIエージェント講義では、このモデルを
# MAGIC **複数のツールと連携させたエージェントシステム**に統合します！
