# Databricks notebook source
# MAGIC %md
# MAGIC # Exercise 1: 基本的なLLM呼び出し
# MAGIC
# MAGIC ## 目的
# MAGIC - Databricks Foundation Model APIの基本的な使い方を理解する
# MAGIC - Chat Completion APIの基本構造を学ぶ
# MAGIC - システムプロンプトとユーザープロンプトの役割を実験的に理解する
# MAGIC
# MAGIC ## 使用するモデル
# MAGIC - **Meta Llama 3.3 70B Instruct**: 高性能な会話型モデル
# MAGIC - Databricks-hosted foundation modelとして提供

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ: 必要なライブラリのインストール

# COMMAND ----------

# DBTITLE 1,ライブラリのインストールと確認
# MAGIC %pip install --upgrade databricks-sdk openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

MODEL_NAME = "databricks-llama-4-maverick"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: シンプルな質問応答
# MAGIC
# MAGIC 最もシンプルな形で、LLMに質問を投げかけてみます。

# COMMAND ----------

# DBTITLE 1,Databricks SDKを使った基本的なクエリ
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# WorkspaceClientの初期化（Notebook内では自動的に認証される）
w = WorkspaceClient()

# OpenAI互換クライアントの取得
openai_client = w.serving_endpoints.get_open_ai_client()

# 基本的な質問
response = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "機械学習とは何ですか？3文で簡潔に説明してください。"
        }
    ],
    max_tokens=256,
    temperature=0.7
)

# 結果の表示
print("=" * 60)
print("【質問】")
print("機械学習とは何ですか？3文で簡潔に説明してください。")
print("\n【LLMの回答】")
print(response.choices[0].message.content)
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 ポイント
# MAGIC - `model`: 使用するモデルのエンドポイント名を指定
# MAGIC - `messages`: 会話履歴を配列で渡す（OpenAI Chat Completion API互換）
# MAGIC - `max_tokens`: 生成する最大トークン数
# MAGIC - `temperature`: 生成のランダム性（0.0=決定的、1.0=創造的）

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: システムプロンプトの効果を理解する
# MAGIC
# MAGIC システムプロンプトは、LLMに「役割」や「振る舞い」を指示する強力な機能です。

# COMMAND ----------

# DBTITLE 1,システムプロンプトなしの回答
response_no_system = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "過学習について説明してください"
        }
    ],
    max_tokens=200,
    temperature=0.5
)

print("【システムプロンプトなし】")
print(response_no_system.choices[0].message.content)
print("\n" + "=" * 60 + "\n")

# COMMAND ----------

# DBTITLE 1,システムプロンプトありの回答（専門家モード）
response_expert = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "あなたは日本の大阪出身の機械学習の専門家です。関西弁を使って説明してください。"
        },
        {
            "role": "user",
            "content": "過学習について説明してください"
        }
    ],
    max_tokens=200,
    temperature=0.5
)

print("【システムプロンプト: 専門家モード】")
print(response_expert.choices[0].message.content)
print("\n" + "=" * 60 + "\n")

# COMMAND ----------

# DBTITLE 1,システムプロンプトありの回答（初心者向けモード）
response_beginner = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "あなたは優しい先生です。機械学習の初心者にもわかるように、専門用語を避けて、日常的な例え話を使って説明してください。"
        },
        {
            "role": "user",
            "content": "過学習について説明してください"
        }
    ],
    max_tokens=200,
    temperature=0.5
)

print("【システムプロンプト: 初心者向けモード】")
print(response_beginner.choices[0].message.content)
print("\n" + "=" * 60 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 観察ポイント
# MAGIC 同じ質問でも、システムプロンプトによって：
# MAGIC - **説明のスタイル**が変化する
# MAGIC - **使用する語彙**が変化する
# MAGIC - **説明の深さ**が変化する

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Temperatureパラメータの効果を理解する
# MAGIC
# MAGIC Temperatureは生成のランダム性を制御します。

# COMMAND ----------

# DBTITLE 1,低Temperature（決定的な回答）
print("【Temperature = 0.0（決定的）】\n")

for i in range(3):
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "「機械学習」を一言で表現してください"
            }
        ],
        max_tokens=50,
        temperature=0.0  # 決定的
    )
    print(f"試行 {i+1}: {response.choices[0].message.content}")

# COMMAND ----------

# DBTITLE 1,高Temperature（創造的な回答）
print("\n【Temperature = 1.5（創造的）】\n")

for i in range(3):
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "「機械学習」を一言で表現してください"
            }
        ],
        max_tokens=50,
        temperature=1.5  # 創造的
    )
    print(f"試行 {i+1}: {response.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 Temperatureの使い分け
# MAGIC - **0.0-0.3**: 事実ベースの回答、コード生成、分類タスク
# MAGIC - **0.7-1.0**: 一般的な会話、要約、説明
# MAGIC - **1.0-2.0**: クリエイティブな文章生成、ブレインストーミング

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: マルチターン会話の実装
# MAGIC
# MAGIC 会話履歴を保持することで、文脈を理解した対話が可能になります。

# COMMAND ----------

# DBTITLE 1,会話履歴を保持した対話
# 会話履歴を保存するリスト
conversation_history = [
    {
        "role": "system",
        "content": "あなたはデータサイエンスの教師です。学生の質問に丁寧に答えてください。"
    }
]

def chat(user_message):
    """会話履歴を保持しながらLLMと対話する関数"""
    # ユーザーメッセージを履歴に追加
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    # LLMに問い合わせ
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation_history,
        max_tokens=300,
        temperature=0.7
    )
    
    # アシスタントの回答を履歴に追加
    assistant_message = response.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

# 会話の実行
print("=" * 60)
print("【ターン1】")
user_msg_1 = "決定木アルゴリズムの利点を教えてください"
print(f"ユーザー: {user_msg_1}")
assistant_msg_1 = chat(user_msg_1)
print(f"アシスタント: {assistant_msg_1}")

print("\n" + "=" * 60)
print("【ターン2】")
user_msg_2 = "それでは、欠点は何ですか？"  # 「それ」= 決定木を参照
print(f"ユーザー: {user_msg_2}")
assistant_msg_2 = chat(user_msg_2)
print(f"アシスタント: {assistant_msg_2}")

print("\n" + "=" * 60)
print("【ターン3】")
user_msg_3 = "その欠点を克服する方法はありますか？"
print(f"ユーザー: {user_msg_3}")
assistant_msg_3 = chat(user_msg_3)
print(f"アシスタント: {assistant_msg_3}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 文脈の保持
# MAGIC - ターン2の「それ」が「決定木」を指していることをLLMが理解している
# MAGIC - ターン3の「その欠点」が前の回答の内容を参照している
# MAGIC - これがChat Completion APIの強力な機能

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: トークン使用量の確認
# MAGIC
# MAGIC 本番運用では、コスト管理のためトークン使用量の監視が重要です。

# COMMAND ----------

# DBTITLE 1,トークン使用量の詳細表示
response_with_usage = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "あなたは簡潔に答えるアシスタントです。"
        },
        {
            "role": "user",
            "content": "Transformerアーキテクチャの主要な構成要素を箇条書きで教えてください"
        }
    ],
    max_tokens=300,
    temperature=0.5
)

# トークン使用量の表示
usage = response_with_usage.usage
print("【トークン使用量】")
print(f"入力トークン数: {usage.prompt_tokens}")
print(f"出力トークン数: {usage.completion_tokens}")
print(f"合計トークン数: {usage.total_tokens}")
print("\n【生成された回答】")
print(response_with_usage.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Exercise 1のまとめ
# MAGIC
# MAGIC このExerciseで学んだこと：
# MAGIC 1. **Databricks Foundation Model APIの基本的な使い方**
# MAGIC    - WorkspaceClientとOpenAI互換クライアントの初期化
# MAGIC    - モデルエンドポイント名の指定方法
# MAGIC
# MAGIC 2. **システムプロンプトの重要性**
# MAGIC    - LLMの振る舞いや口調を制御できる
# MAGIC    - タスクに応じて専門性のレベルを調整できる
# MAGIC
# MAGIC 3. **Temperatureパラメータの効果**
# MAGIC    - 決定的な回答（低Temperature）vs 創造的な回答（高Temperature）
# MAGIC    - タスクに応じた適切な値の選択
# MAGIC
# MAGIC 4. **マルチターン会話の実装**
# MAGIC    - 会話履歴を保持することで文脈を理解した対話が可能
# MAGIC    - messagesリストの管理方法
# MAGIC
# MAGIC 5. **トークン使用量の監視**
# MAGIC    - コスト管理とパフォーマンス最適化の基礎

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 課題（オプション）
# MAGIC
# MAGIC 時間があれば、以下の実験をしてみてください：
# MAGIC
# MAGIC 1. **異なるシステムプロンプトを試す**
# MAGIC    - 「あなたは詩人です」「あなたはコメディアンです」など
# MAGIC
# MAGIC 2. **max_tokensの影響を観察する**
# MAGIC    - 50, 100, 500と変えて、回答の長さと質の関係を確認
# MAGIC
# MAGIC 3. **別のモデルを試す**
# MAGIC    - `databricks-gemini-2-5-pro`など他のモデルと比較
# MAGIC
# MAGIC 4. **実用的なタスクを設計する**
# MAGIC    - ビジネスメールの下書き作成
# MAGIC    - コードのバグ解説
# MAGIC    - データ分析結果の要約
