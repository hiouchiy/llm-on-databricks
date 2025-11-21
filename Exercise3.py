# Databricks notebook source
# MAGIC %md
# MAGIC # Exercise 3: Function Callingの基礎
# MAGIC
# MAGIC ## 目的
# MAGIC - LLMが外部ツール/APIを呼び出す仕組みを理解する
# MAGIC - `tools`パラメータと`tool_choice`の使い方を学ぶ
# MAGIC - マルチターン会話でのツール実行フローを実装する
# MAGIC - 次回のAIエージェント講義への基盤を構築する
# MAGIC
# MAGIC ## ビジネス背景
# MAGIC カスタマーサポートボットやバーチャルアシスタントでは、LLMが以下のような外部システムと連携する必要があります：
# MAGIC - **天気情報API**: 旅行予約や配送計画の相談
# MAGIC - **在庫確認システム**: 商品の在庫状況確認
# MAGIC - **データベース検索**: 注文履歴や顧客情報の取得
# MAGIC
# MAGIC Function Callingは、これらの連携を実現する基盤技術です。

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install --upgrade databricks-sdk openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

MODEL_NAME = "databricks-llama-4-maverick"

# COMMAND ----------

# DBTITLE 1,クライアントの初期化
from databricks.sdk import WorkspaceClient
from openai import OpenAI
import json
from typing import Literal

# WorkspaceClientの初期化
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

print("✅ クライアントの初期化が完了しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: シンプルなFunction Calling - 天気情報取得
# MAGIC
# MAGIC まず、1つのツールを定義して、LLMがそれを呼び出す様子を観察します。

# COMMAND ----------

# DBTITLE 1,ツール定義: 天気情報取得API
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "指定された都市の現在の天気情報を取得します。気温、天候、湿度などの情報が得られます。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "都市名（例: 東京, 大阪, ニューヨーク）"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度の単位"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

print("【定義したツール】")
print(json.dumps(tools, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 ツール定義のポイント
# MAGIC - `name`: ツールの識別名（関数名に対応）
# MAGIC - `description`: LLMがツールを選択する際の判断材料となる重要な説明
# MAGIC - `parameters`: JSON Schemaで引数を定義
# MAGIC - `required`: 必須引数のリスト

# COMMAND ----------

# DBTITLE 1,LLMにツール呼び出しを依頼
user_query = "東京の天気を教えてください"

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": user_query
        }
    ],
    tools=tools,
    tool_choice="auto"  # LLMが自動的にツールを呼ぶかどうか判断
)

# レスポンスの確認
message = response.choices[0].message

print("【ユーザーの質問】")
print(user_query)
print("\n【LLMの応答】")
print(f"Finish Reason: {response.choices[0].finish_reason}")

if message.tool_calls:
    print("\n✅ LLMがツールを呼び出しました！")
    for tool_call in message.tool_calls:
        print(f"\nツール名: {tool_call.function.name}")
        print(f"引数: {tool_call.function.arguments}")
        
        # JSON形式で整形表示
        args = json.loads(tool_call.function.arguments)
        print(f"\n【パースされた引数】")
        print(json.dumps(args, indent=2, ensure_ascii=False))
else:
    print("\n⚠️ ツールは呼び出されませんでした")
    print(f"通常の応答: {message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 観察ポイント
# MAGIC - LLMは「東京の天気」という自然言語を理解し、`get_current_weather`関数を呼び出すことを決定
# MAGIC - 引数`location`に「東京」を自動的に抽出
# MAGIC - `unit`は指定されていないため、デフォルトまたは省略される可能性がある

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: 実際のツール関数を実装する
# MAGIC
# MAGIC LLMは関数を実際には実行しません。開発者が実装する必要があります。

# COMMAND ----------

# DBTITLE 1,モック天気APIの実装
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """
    天気情報を取得する関数（モック実装）
    実際のプロダクションでは、OpenWeatherMap APIなどを呼び出す
    """
    # モックデータ
    weather_data = {
        "東京": {"temperature": 15, "condition": "曇り", "humidity": 65},
        "大阪": {"temperature": 17, "condition": "晴れ", "humidity": 55},
        "ニューヨーク": {"temperature": 8, "condition": "雨", "humidity": 75},
        "ロンドン": {"temperature": 10, "condition": "曇り", "humidity": 80},
        "シドニー": {"temperature": 22, "condition": "快晴", "humidity": 50},
    }
    
    # 都市名の正規化（部分一致）
    matched_city = None
    for city in weather_data.keys():
        if city in location or location in city:
            matched_city = city
            break
    
    if matched_city:
        data = weather_data[matched_city]
        temp = data["temperature"]
        
        # 華氏変換
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
        
        return {
            "location": matched_city,
            "temperature": temp,
            "unit": unit,
            "condition": data["condition"],
            "humidity": data["humidity"]
        }
    else:
        return {
            "error": f"都市'{location}'の天気情報は見つかりませんでした"
        }

# テスト
test_result = get_current_weather("東京", "celsius")
print("【テスト実行】")
print(json.dumps(test_result, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: マルチターン会話 - ツール実行結果をLLMに返す
# MAGIC
# MAGIC Function Callingの完全なフローを実装します。

# COMMAND ----------

# DBTITLE 1,完全なFunction Callingフローの実装
def run_conversation(user_query: str) -> str:
    """
    ユーザーの質問に対して、必要に応じてツールを呼び出し、最終回答を返す
    """
    # 会話履歴
    messages = [{"role": "user", "content": user_query}]
    
    print(f"{'='*60}")
    print(f"【ユーザーの質問】\n{user_query}")
    print(f"{'='*60}\n")
    
    # Step 1: LLMに問い合わせ（ツール定義付き）
    print("Step 1: LLMにツールを提示して問い合わせ中...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # Step 2: ツール呼び出しの確認と実行
    if response_message.tool_calls:
        print("✅ LLMがツールの呼び出しを要求しました\n")
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"【ツール呼び出し】")
            print(f"関数名: {function_name}")
            print(f"引数: {json.dumps(function_args, ensure_ascii=False)}")
            
            # Step 3: 実際の関数を実行
            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "celsius")
                )
            else:
                function_response = {"error": "Unknown function"}
            
            print(f"【ツールの実行結果】")
            print(json.dumps(function_response, indent=2, ensure_ascii=False))
            
            # Step 4: ツールの実行結果を会話履歴に追加
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False)
            })
        
        # Step 5: ツール実行結果を含めて再度LLMに問い合わせ
        print("\nStep 2: ツール実行結果をLLMに渡して最終回答を生成中...\n")
        second_response = client.chat.completions.create(
            model="databricks-meta-llama-3-3-70b-instruct",
            messages=messages
        )
        
        final_answer = second_response.choices[0].message.content
        
    else:
        # ツール呼び出しなし
        print("⚠️ ツールは使用されませんでした\n")
        final_answer = response_message.content
    
    print(f"{'='*60}")
    print(f"【最終回答】")
    print(final_answer)
    print(f"{'='*60}\n")
    
    return final_answer

# テスト実行
answer1 = run_conversation("東京の天気を教えてください")

# COMMAND ----------

# DBTITLE 1,別の質問でテスト
answer2 = run_conversation("ニューヨークは今何度ですか？華氏で教えてください")

# COMMAND ----------

# DBTITLE 1,ツールが不要な質問でテスト
answer3 = run_conversation("機械学習とは何ですか？")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 マルチターンフローの理解
# MAGIC
# MAGIC 1. **ユーザー → LLM**: 質問を投げる（ツール定義付き）
# MAGIC 2. **LLM → 開発者**: ツール呼び出し要求（JSON形式）
# MAGIC 3. **開発者 → API**: 実際のツール実行
# MAGIC 4. **開発者 → LLM**: 実行結果を`role: tool`メッセージとして返す
# MAGIC 5. **LLM → ユーザー**: 結果を自然言語で要約して回答
# MAGIC
# MAGIC これが**AIエージェント**の基本動作パターンです！

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: 複数のツールを定義する
# MAGIC
# MAGIC 実際のアプリケーションでは、複数のツールから適切なものをLLMが選択します。

# COMMAND ----------

# DBTITLE 1,複数ツールの定義
multi_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "指定された都市の現在の天気情報を取得します",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "都市名"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度の単位"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "指定された商品の在庫状況を確認します。商品名またはSKUコードで検索できます。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "商品名またはSKUコード"
                    },
                    "warehouse": {
                        "type": "string",
                        "enum": ["tokyo", "osaka", "nagoya"],
                        "description": "倉庫の場所（省略可）"
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping_cost",
            "description": "配送料金を計算します。発送元、配送先、重量から料金を算出します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_location": {
                        "type": "string",
                        "description": "発送元の都市名"
                    },
                    "to_location": {
                        "type": "string",
                        "description": "配送先の都市名"
                    },
                    "weight_kg": {
                        "type": "number",
                        "description": "荷物の重量（キログラム）"
                    }
                },
                "required": ["from_location", "to_location", "weight_kg"]
            }
        }
    }
]

print(f"✅ {len(multi_tools)}個のツールを定義しました")

# COMMAND ----------

# DBTITLE 1,追加ツールの実装
def check_inventory(product_name: str, warehouse: str = None) -> dict:
    """在庫確認（モック実装）"""
    inventory_data = {
        "ノートパソコン": {"tokyo": 15, "osaka": 8, "nagoya": 3},
        "ワイヤレスマウス": {"tokyo": 50, "osaka": 30, "nagoya": 20},
        "Bluetoothイヤホン": {"tokyo": 25, "osaka": 35, "nagoya": 10},
        "モニター": {"tokyo": 10, "osaka": 5, "nagoya": 2}
    }
    
    # 部分一致検索
    matched_product = None
    for product in inventory_data.keys():
        if product in product_name or product_name in product:
            matched_product = product
            break
    
    if matched_product:
        if warehouse:
            stock = inventory_data[matched_product].get(warehouse, 0)
            return {
                "product": matched_product,
                "warehouse": warehouse,
                "stock": stock,
                "status": "在庫あり" if stock > 0 else "在庫なし"
            }
        else:
            return {
                "product": matched_product,
                "inventory_by_warehouse": inventory_data[matched_product],
                "total_stock": sum(inventory_data[matched_product].values())
            }
    else:
        return {"error": f"商品'{product_name}'が見つかりません"}

def calculate_shipping_cost(from_location: str, to_location: str, weight_kg: float) -> dict:
    """配送料金計算（モック実装）"""
    # 基本料金
    base_rate = 500
    
    # 距離係数（簡易計算）
    distance_multiplier = 1.0
    if "東京" in from_location or "東京" in to_location:
        if "大阪" in from_location or "大阪" in to_location:
            distance_multiplier = 1.5
        elif "札幌" in from_location or "札幌" in to_location:
            distance_multiplier = 2.0
    
    # 重量による追加料金
    weight_charge = weight_kg * 100
    
    # 合計
    total_cost = int(base_rate * distance_multiplier + weight_charge)
    
    return {
        "from": from_location,
        "to": to_location,
        "weight_kg": weight_kg,
        "shipping_cost_jpy": total_cost,
        "estimated_days": 2 if distance_multiplier < 2 else 3
    }

print("✅ 追加ツール関数を実装しました")

# COMMAND ----------

# DBTITLE 1,マルチツール対応の会話関数
def run_multi_tool_conversation(user_query: str) -> str:
    """複数ツールに対応した会話実行"""
    messages = [{"role": "user", "content": user_query}]
    
    print(f"{'='*60}")
    print(f"【ユーザーの質問】\n{user_query}")
    print(f"{'='*60}\n")
    
    # LLMに問い合わせ
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=multi_tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # ツール呼び出しの処理
    if response_message.tool_calls:
        print("✅ LLMがツールの呼び出しを要求しました\n")
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"【ツール: {function_name}】")
            print(f"引数: {json.dumps(function_args, ensure_ascii=False)}")
            
            # 適切な関数を実行
            if function_name == "get_current_weather":
                function_response = get_current_weather(**function_args)
            elif function_name == "check_inventory":
                function_response = check_inventory(**function_args)
            elif function_name == "calculate_shipping_cost":
                function_response = calculate_shipping_cost(**function_args)
            else:
                function_response = {"error": "Unknown function"}
            
            print(f"実行結果: {json.dumps(function_response, indent=2, ensure_ascii=False)}\n")
            
            # 結果を会話履歴に追加
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False)
            })
        
        # 最終回答生成
        second_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        final_answer = second_response.choices[0].message.content
    else:
        final_answer = response_message.content
    
    print(f"{'='*60}")
    print(f"【最終回答】")
    print(final_answer)
    print(f"{'='*60}\n")
    
    return final_answer

# COMMAND ----------

# DBTITLE 1,様々な質問でテスト
# テスト1: 在庫確認
run_multi_tool_conversation("ノートパソコンの在庫を教えてください")

# COMMAND ----------

# テスト2: 配送料金計算
run_multi_tool_conversation("東京から大阪まで5kgの荷物を送るといくらですか？")

# COMMAND ----------

# テスト3: 複合的な質問
run_multi_tool_conversation("大阪倉庫のワイヤレスマウスの在庫はありますか？")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 LLMのツール選択能力
# MAGIC
# MAGIC 注目すべきポイント：
# MAGIC - LLMは質問内容から**適切なツールを自動選択**している
# MAGIC - 複数ツールがあっても、混乱せずに正しいツールを呼び出す
# MAGIC - これはツールの`description`が明確に書かれているため
# MAGIC
# MAGIC **良いdescriptionの書き方**：
# MAGIC - 具体的な使用例を含める
# MAGIC - ツールが「何をするか」だけでなく「いつ使うべきか」も説明
# MAGIC - 他のツールとの違いを明確にする

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: tool_choiceパラメータの制御
# MAGIC
# MAGIC ツールの呼び出し方を制御できます。

# COMMAND ----------

# DBTITLE 1,tool_choiceの動作比較
test_query = "こんにちは、何か手伝えることはありますか？"

print("【テスト質問】")
print(test_query)
print("\n" + "="*60 + "\n")

# 1. auto（デフォルト）: LLMが自動判断
print("1️⃣ tool_choice='auto'")
response_auto = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": test_query}],
    tools=multi_tools,
    tool_choice="auto"
)
print(f"Finish Reason: {response_auto.choices[0].finish_reason}")
if response_auto.choices[0].message.tool_calls:
    print("→ ツールを呼び出しました")
else:
    print("→ ツールを呼び出しませんでした")
    print(f"応答: {response_auto.choices[0].message.content}")

print("\n" + "="*60 + "\n")

# 2. required: 必ずツールを呼び出す
print("2️⃣ tool_choice='required'")
response_required = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": test_query}],
    tools=multi_tools,
    tool_choice="required"
)
print(f"Finish Reason: {response_required.choices[0].finish_reason}")
if response_required.choices[0].message.tool_calls:
    print("→ ツールを呼び出しました")
    for tc in response_required.choices[0].message.tool_calls:
        print(f"   {tc.function.name}")
else:
    print("→ ツールを呼び出しませんでした")

print("\n" + "="*60 + "\n")

# 3. none: ツールを呼び出さない
print("3️⃣ tool_choice='none'")
response_none = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": test_query}],
    tools=multi_tools,
    tool_choice="none"
)
print(f"Finish Reason: {response_none.choices[0].finish_reason}")
if response_none.choices[0].message.tool_calls:
    print("→ ツールを呼び出しました")
else:
    print("→ ツールを呼び出しませんでした")
    print(f"応答: {response_none.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 tool_choiceの使い分け
# MAGIC
# MAGIC - **`auto`**: 通常のチャットボット（LLMの判断に任せる）
# MAGIC - **`required`**: 必ず何らかのアクションを実行させたい場合（データ抽出パイプラインなど）
# MAGIC - **`none`**: ツール呼び出しを一時的に無効化（通常の会話モード）
# MAGIC - **特定関数の指定**: 特定のワークフローで決まったツールのみ使用

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Exercise 3のまとめ
# MAGIC
# MAGIC このExerciseで学んだこと：
# MAGIC
# MAGIC ### 技術面
# MAGIC 1. **Function Callingの仕組み**
# MAGIC    - LLMは関数を実際には実行しない（JSONを生成するだけ）
# MAGIC    - 開発者が実際のツール実行とLLMへの結果返却を担当
# MAGIC    - `role: tool`メッセージで結果をLLMに渡す
# MAGIC
# MAGIC 2. **ツール定義のベストプラクティス**
# MAGIC    - 明確で具体的な`description`を書く
# MAGIC    - JSON Schemaで引数の型と制約を定義
# MAGIC    - `required`で必須引数を明示
# MAGIC
# MAGIC 3. **マルチターン会話フロー**
# MAGIC    - ユーザー → LLM → ツール実行 → LLM → ユーザー
# MAGIC    - 会話履歴の管理が重要
# MAGIC
# MAGIC 4. **tool_choiceによる制御**
# MAGIC    - `auto`, `required`, `none`の使い分け
# MAGIC    - 特定関数を強制的に呼び出す方法
# MAGIC
# MAGIC ### ビジネス面
# MAGIC 5. **実用的なユースケース**
# MAGIC    - カスタマーサポートボット（在庫確認、配送料金計算）
# MAGIC    - バーチャルアシスタント（天気情報、スケジュール管理）
# MAGIC    - 社内ヘルプデスク（ドキュメント検索、システム操作）

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 次回講義への橋渡し
# MAGIC
# MAGIC 今回学んだFunction Callingは、**AIエージェント**の基盤技術です。
# MAGIC
# MAGIC ### 今回 vs 次回
# MAGIC
# MAGIC **今回（Exercise 3）**:
# MAGIC - 単一ツールの呼び出し
# MAGIC - 開発者が明示的に会話フローを制御
# MAGIC - ツール実行は1回のみ
# MAGIC
# MAGIC **次回（AIエージェント講義）**:
# MAGIC - 複数ツールの連鎖的な呼び出し
# MAGIC - LangGraphなどのフレームワークによる自動制御
# MAGIC - ツールの実行結果に基づいて次のツールを決定
# MAGIC - エラーハンドリングと再試行
# MAGIC - メモリとコンテキストの管理
# MAGIC
# MAGIC ### 発展トピック
# MAGIC - **Parallel Function Calling**: 複数ツールを並列実行
# MAGIC - **Tool as Agent**: ツール自体が別のLLMを呼び出す
# MAGIC - **Human-in-the-Loop**: 重要な操作は人間の承認を得る
# MAGIC - **RAG (Retrieval-Augmented Generation)**: ツールとしての知識ベース検索

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 発展課題（オプション）
# MAGIC
# MAGIC 時間があれば、以下にチャレンジしてください：
# MAGIC
# MAGIC 1. **新しいツールを追加**
# MAGIC    - `send_email(to, subject, body)`: メール送信
# MAGIC    - `create_calendar_event(title, date, time)`: カレンダー登録
# MAGIC    - `search_documents(query)`: 社内ドキュメント検索
# MAGIC
# MAGIC 2. **エラーハンドリング**
# MAGIC    - ツール実行が失敗した場合の処理
# MAGIC    - LLMに再試行を促す
# MAGIC
# MAGIC 3. **複数ツール連鎖**
# MAGIC    - 「東京の天気を確認して、雨なら傘を注文」のような複合タスク
# MAGIC
# MAGIC 4. **実際のAPIとの統合**
# MAGIC    - OpenWeatherMap APIを実際に呼び出す
# MAGIC    - Databricks SQLでデータベースクエリを実行
