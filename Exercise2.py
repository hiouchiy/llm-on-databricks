# Databricks notebook source
# MAGIC %md
# MAGIC # Exercise 2: Structured Outputsによるデータ抽出
# MAGIC
# MAGIC ## 注意事項
# MAGIC Databricks Foundation Model APIの`strict: True`モードでは、以下の制約はサポートされていません：
# MAGIC - 数値の `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`
# MAGIC - 配列の `minItems`, `maxItems`
# MAGIC - 文字列の `minLength`, `maxLength`
# MAGIC
# MAGIC これらの制約は**description**に記載し、LLMに自然言語で指示します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install --upgrade databricks-sdk openai pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

MODEL_NAME = "databricks-llama-4-maverick"

# COMMAND ----------

# DBTITLE 1,クライアントの初期化
from databricks.sdk import WorkspaceClient
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
import json

# WorkspaceClientの初期化
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

print("✅ クライアントの初期化が完了しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: シンプルな構造化出力

# COMMAND ----------

# DBTITLE 1,Pydanticモデルの定義（制約なし版）
class SimpleReview(BaseModel):
    """シンプルなレビュー構造"""
    product_name: str = Field(description="製品名")
    rating: int = Field(description="評価スコア（1から5の整数値、5が最高評価）")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="全体的な感情"
    )

# スキーマの確認
print("【定義したJSONスキーマ】")
print(json.dumps(SimpleReview.model_json_schema(), indent=2, ensure_ascii=False))

# COMMAND ----------

# DBTITLE 1,レビューテキストの準備
sample_review_1 = """
このBluetoothイヤホン、最高です！
音質も良いし、バッテリーも1日持ちます。
通勤時間が快適になりました。5つ星です！
"""

print("【入力レビューテキスト】")
print(sample_review_1)

# COMMAND ----------

# DBTITLE 1,Structured Outputsで情報抽出
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": """あなたは顧客レビューから情報を抽出する専門家です。
以下のルールに従ってください：
- ratingは必ず1から5の整数値で、1が最低、5が最高です
- レビュー内容に基づいて正確に判定してください"""
        },
        {
            "role": "user",
            "content": f"以下のレビューから情報を抽出してください:\n\n{sample_review_1}"
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "simple_review",
            "schema": SimpleReview.model_json_schema(),
            "strict": True
        }
    },
    temperature=0.0
)

# 結果の取得とパース
extracted_data = json.loads(response.choices[0].message.content)

print("\n【抽出された構造化データ】")
print(json.dumps(extracted_data, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: 複雑な構造化出力

# COMMAND ----------

# DBTITLE 1,詳細なレビュー構造の定義（制約なし版）
class ProductFeature(BaseModel):
    """製品の個別機能の評価"""
    feature_name: str = Field(description="機能名（例：音質、バッテリー、デザイン）")
    mentioned: bool = Field(description="この機能がレビューで言及されているか")
    rating: int = Field(description="この機能の評価（1-5の整数、5が最高）")
    comment: str = Field(description="この機能に関するコメントの簡潔な要約")

class DetailedReview(BaseModel):
    """詳細なレビュー分析"""
    product_name: str = Field(description="製品名")
    overall_rating: int = Field(description="総合評価（1-5の整数、5が最高）")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="全体的な感情"
    )
    features: List[ProductFeature] = Field(
        description="言及された製品機能のリスト"
    )
    key_issues: List[str] = Field(
        description="主要な問題点や不満のリスト（なければ空リスト）"
    )
    key_praises: List[str] = Field(
        description="主要な長所や称賛のリスト"
    )
    would_recommend: bool = Field(
        description="購入を推奨するか"
    )
    urgency_level: Literal["low", "medium", "high"] = Field(
        description="カスタマーサポートの対応優先度（low=満足/medium=改善希望/high=返品希望や使用不可）"
    )

print("【詳細なJSONスキーマを定義しました】")

# COMMAND ----------

# DBTITLE 1,複数のレビューサンプル
sample_reviews = [
    {
        "id": "R001",
        "text": """
        ワイヤレスマウスを購入しましたが、接続が頻繁に切れて使い物になりません。
        デザインは良いのですが、肝心の機能が全くダメです。
        電池の持ちも悪く、2週間で3回も交換しました。
        すぐに返品したいです。星1つです。
        """
    },
    {
        "id": "R002",
        "text": """
        このノートパソコン、素晴らしいです！
        画面が綺麗で目が疲れにくく、長時間の作業も快適です。
        バッテリーは1回の充電で10時間持ちます。
        少し重いのが難点ですが、性能を考えれば許容範囲です。
        プログラミングにもビデオ編集にも最適。5つ星です！
        """
    },
    {
        "id": "R003",
        "text": """
        Bluetoothスピーカーを買いました。
        音質は価格の割には普通です。可もなく不可もなく。
        防水機能は便利ですが、音量が少し小さいかな。
        まあ、この値段なら妥当だと思います。
        """
    }
]

print(f"【{len(sample_reviews)}件のレビューサンプルを準備しました】")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: バッチ処理で複数レビューを分析

# COMMAND ----------

# DBTITLE 1,レビュー分析関数の実装
def analyze_review(review_text: str, review_id: str) -> dict:
    """
    レビューテキストを構造化データに変換する
    
    Args:
        review_text: レビューの生テキスト
        review_id: レビューID
    
    Returns:
        構造化されたレビューデータ
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """あなたはEコマースプラットフォームのレビュー分析専門家です。
顧客レビューから詳細な情報を抽出し、以下の観点で分析してください：
- 製品の各機能に対する評価（ratingは1-5の整数、5が最高）
- 具体的な問題点と長所
- カスタマーサポートの対応優先度（緊急度）

評価スコアのルール：
- overall_ratingとfeature ratingは必ず1から5の整数値
- 1=非常に悪い、2=悪い、3=普通、4=良い、5=非常に良い

緊急度の判断基準：
- high: 返品希望、製品が使えない、安全上の問題
- medium: 改善してほしい点がある、期待と異なる
- low: 満足している、軽微な不満のみ"""
            },
            {
                "role": "user",
                "content": f"以下のレビューを分析してください:\n\n{review_text}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "detailed_review_analysis",
                "schema": DetailedReview.model_json_schema(),
                "strict": True
            }
        },
        temperature=0.0
    )
    
    # JSONパース
    result = json.loads(response.choices[0].message.content)
    result["review_id"] = review_id  # IDを追加
    
    return result

print("✅ レビュー分析関数を定義しました")

# COMMAND ----------

# DBTITLE 1,全レビューを処理
analyzed_reviews = []

for review in sample_reviews:
    print(f"\n{'='*60}")
    print(f"処理中: レビューID {review['id']}")
    print(f"{'='*60}")
    
    result = analyze_review(review['text'], review['id'])
    analyzed_reviews.append(result)
    
    # 結果の表示
    print(f"\n製品名: {result['product_name']}")
    print(f"総合評価: {result['overall_rating']}/5 ({'⭐' * result['overall_rating']})")
    print(f"感情: {result['sentiment']}")
    print(f"推奨度: {'はい' if result['would_recommend'] else 'いいえ'}")
    print(f"緊急度: {result['urgency_level'].upper()}")
    
    print(f"\n【長所】")
    for praise in result['key_praises']:
        print(f"  ✓ {praise}")
    
    print(f"\n【問題点】")
    if result['key_issues']:
        for issue in result['key_issues']:
            print(f"  ✗ {issue}")
    else:
        print("  （なし）")
    
    print(f"\n【機能評価】")
    for feature in result['features']:
        status = "✓" if feature['mentioned'] else "−"
        print(f"  {status} {feature['feature_name']}: {feature['rating']}/5")
        print(f"     → {feature['comment']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: DataFrameに変換してビジネス分析

# COMMAND ----------

# DBTITLE 1,DataFrameへの変換
import pandas as pd

# メインデータの抽出
df_reviews = pd.DataFrame([
    {
        "review_id": r["review_id"],
        "product_name": r["product_name"],
        "overall_rating": r["overall_rating"],
        "sentiment": r["sentiment"],
        "would_recommend": r["would_recommend"],
        "urgency_level": r["urgency_level"],
        "num_issues": len(r["key_issues"]),
        "num_praises": len(r["key_praises"]),
        "num_features": len(r["features"])
    }
    for r in analyzed_reviews
])

print("【レビューデータフレーム】")
display(df_reviews)

# COMMAND ----------

# DBTITLE 1,集計分析
print("【ビジネス分析サマリー】\n")

print("1. 平均評価スコア")
print(f"   {df_reviews['overall_rating'].mean():.2f} / 5.0")

print("\n2. 感情分布")
sentiment_dist = df_reviews['sentiment'].value_counts()
for sentiment, count in sentiment_dist.items():
    print(f"   {sentiment}: {count}件 ({count/len(df_reviews)*100:.1f}%)")

print("\n3. 推奨率")
recommend_rate = df_reviews['would_recommend'].sum() / len(df_reviews) * 100
print(f"   {recommend_rate:.1f}%")

print("\n4. 緊急対応が必要なレビュー")
urgent_reviews = df_reviews[df_reviews['urgency_level'] == 'high']
print(f"   {len(urgent_reviews)}件 / {len(df_reviews)}件")
if len(urgent_reviews) > 0:
    print(f"   対象レビューID: {', '.join(urgent_reviews['review_id'].tolist())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: バリデーション関数の実装

# COMMAND ----------

# DBTITLE 1,カスタムバリデーション
def validate_extracted_data(json_data: dict) -> tuple[bool, str]:
    """
    抽出されたデータをバリデーションする
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # 評価スコアの範囲チェック
        if not (1 <= json_data['overall_rating'] <= 5):
            return False, f"overall_ratingが範囲外: {json_data['overall_rating']}"
        
        # 機能評価の範囲チェック
        for feature in json_data['features']:
            if not (1 <= feature['rating'] <= 5):
                return False, f"feature ratingが範囲外: {feature['rating']}"
        
        # ビジネスルールの検証
        if json_data['overall_rating'] <= 2 and json_data['sentiment'] == 'positive':
            return False, "評価が低いのに感情がpositiveです"
        
        if json_data['overall_rating'] >= 4 and json_data['sentiment'] == 'negative':
            return False, "評価が高いのに感情がnegativeです"
        
        if not json_data['would_recommend'] and json_data['overall_rating'] >= 4:
            return False, "高評価なのに推奨しない矛盾があります"
        
        return True, "OK"
    
    except Exception as e:
        return False, f"バリデーションエラー: {str(e)}"

# 全レビューを検証
print("【データ品質チェック】\n")
for review in analyzed_reviews:
    is_valid, message = validate_extracted_data(review)
    status = "✅" if is_valid else "❌"
    print(f"{status} レビューID {review['review_id']}: {message}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎁 おまけ: Databricksバッチ推論
# MAGIC
# MAGIC ここまでは**リアルタイム推論**（1件ずつ処理）を行ってきましたが、
# MAGIC 実務では**大量データを一括処理**する必要があることが多いです。
# MAGIC
# MAGIC Databricksでは以下の方法でバッチ推論が可能です：
# MAGIC 1. **ai_query() 関数**: Spark DataFrameに対してLLM推論を適用
# MAGIC 2. **Structured Streaming**: 準リアルタイムでの継続的処理

# COMMAND ----------

CATALOG = "handson"
SCHEMA = "llm_lecture"
TABLE = "batch_reviews"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法1: ai_query()を使ったバッチ推論
# MAGIC
# MAGIC `ai_query()`は、SQL/DataFrameで直接LLMを呼び出せる最も簡単な方法です。

# COMMAND ----------

# DBTITLE 1,サンプルデータの準備（Delta Tableとして保存）
import pandas as pd
from pyspark.sql import SparkSession

# 大量のレビューデータを想定
batch_reviews = [
    {"review_id": "B001", "product": "ノートPC", "review_text": "画面が綺麗で作業がはかどります。少し重いですが満足です。"},
    {"review_id": "B002", "product": "マウス", "review_text": "接続が安定しません。すぐに返品したいです。"},
    {"review_id": "B003", "product": "キーボード", "review_text": "打鍵感が最高。タイピングが楽しくなりました。"},
    {"review_id": "B004", "product": "モニター", "review_text": "色が鮮やかで目が疲れにくい。値段も手頃。"},
    {"review_id": "B005", "product": "Webカメラ", "review_text": "画質は普通。マイクの音質が悪いのが残念。"},
    {"review_id": "B006", "product": "スピーカー", "review_text": "音質は価格相応。デザインはおしゃれです。"},
    {"review_id": "B007", "product": "ヘッドセット", "review_text": "長時間つけても痛くない。音質も良好です。"},
    {"review_id": "B008", "product": "外付けSSD", "review_text": "転送速度が速い！容量も十分で大満足。"},
]

# Pandas DataFrameを作成
pdf = pd.DataFrame(batch_reviews)

# Spark DataFrameに変換
df_batch = spark.createDataFrame(pdf)

# Delta Tableとして保存（catalog.schema.table形式に注意）
# Free Editionでは`main.default`スキーマを使用
table_name = f"{CATALOG}.{SCHEMA}.{TABLE}"
df_batch.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✅ {len(batch_reviews)}件のレビューを {table_name} に保存しました")
display(df_batch)

# COMMAND ----------

# DBTITLE 1,ai_query()を使ったバッチ感情分析（SQL）
resultDF = spark.sql(f"""
SELECT 
  review_id,
  product,
  review_text,
  ai_query(
    '{MODEL_NAME}',
    CONCAT(
      'レビューを読んで、感情を「positive」「negative」「neutral」のいずれかで分類してください。単語のみで回答してください。\n\nレビュー: ',
      review_text
    )
  ) AS sentiment
FROM {CATALOG}.{SCHEMA}.{TABLE}
""")

display(resultDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 ai_query()の特徴
# MAGIC
# MAGIC **メリット**:
# MAGIC - SQLで簡単に実行できる（データアナリストにも使いやすい）
# MAGIC - Sparkの分散処理で自動的に並列化
# MAGIC - Delta Tableと統合しやすい
# MAGIC
# MAGIC **制限**:
# MAGIC - Structured Outputsは現在サポートされていない（近日対応予定）
# MAGIC - 複雑なロジックにはPythonの方が向いている

# COMMAND ----------

# DBTITLE 1,ai_query()をPython DataFrameで使用
from pyspark.sql.functions import expr

# Pythonでも同じことができる
df_with_sentiment = df_batch.withColumn(
    "sentiment",
    expr(f"""
        ai_query(
            '{MODEL_NAME}',
            CONCAT(
                'レビューを読んで、感情を「positive」「negative」「neutral」のいずれかで分類してください。単語のみで回答してください。\\n\\nレビュー: ',
                review_text
            )
        )
    """)
)

print("【バッチ推論結果】")
display(df_with_sentiment)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法2: Structured Streamingによる準リアルタイム処理
# MAGIC
# MAGIC 新しいレビューが継続的に追加される場合、Structured Streamingを使用します。

# COMMAND ----------

# DBTITLE 1,Streaming処理の設定（デモ）
from pyspark.sql.functions import expr, col

# Streamingソースとして既存のDelta Tableを読み取る
# 実際には、Kafkaやイベントハブからストリーミングデータを取得
df_stream = spark.readStream.format("delta") .option("maxBytesPerTrigger", "10k").table(table_name)

# ai_query()を適用
df_stream_analyzed = df_stream.withColumn(
    "sentiment",
    expr("""
        ai_query(
            '{MODEL_NAME}',
            CONCAT(
                'レビューを読んで、感情を「positive」「negative」「neutral」のいずれかで分類してください。単語のみで回答してください。\\n\\nレビュー: ',
                review_text
            )
        )
    """)
)

# 結果をDelta Tableに書き込み（実際のストリーミング処理）
# このセルは実行すると継続的に動作するため、デモではコメントアウト
"""
query = df_stream_analyzed.writeStream \
    .format("delta") \
    .option("checkpointLocation", "/tmp/checkpoints/reviews_sentiment") \
    .outputMode("append") \
    .toTable("main.default.reviews_with_sentiment")

query.awaitTermination()
"""

print("⚠️ Streaming処理は継続的に実行されるため、このセルではコード例のみを示しています")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 バッチ推論の使い分け
# MAGIC
# MAGIC | 方法 | 適用ケース | メリット | デメリット |
# MAGIC |------|----------|---------|----------|
# MAGIC | **ai_query() + SQL** | シンプルな感情分析、要約、翻訳 | 実装が簡単、SQLで完結 | Structured Outputs未対応 |
# MAGIC | **Pandas UDF** | 複雑な構造化出力、カスタムロジック | 柔軟性が高い、Structured Outputs対応 | パフォーマンスチューニングが必要 |
# MAGIC | **Structured Streaming** | リアルタイムデータの継続的処理 | 準リアルタイム、スケーラブル | 複雑な設定が必要 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 バッチ推論のベストプラクティス
# MAGIC
# MAGIC 1. **データのパーティショニング**
# MAGIC    - 大規模データは適切にパーティション分割
# MAGIC    - `df.repartition(100)` で並列度を調整
# MAGIC
# MAGIC 2. **エラーハンドリング**
# MAGIC    - UDF内で例外をキャッチし、部分的な失敗を許容
# MAGIC    - 失敗したレコードは別テーブルに記録
# MAGIC
# MAGIC 3. **コスト管理**
# MAGIC    - Foundation Model APIの料金はトークン数に応じて課金
# MAGIC    - `max_tokens`を適切に設定してコストを制御
# MAGIC
# MAGIC 4. **モニタリング**
# MAGIC    - 処理時間、失敗率、トークン消費量をログに記録
# MAGIC    - Delta Tableの履歴機能で処理結果をバージョン管理
# MAGIC
# MAGIC 5. **増分処理**
# MAGIC    - 全データを再処理せず、新規/更新レコードのみ処理
# MAGIC    - `MERGE INTO`を使った効率的なデータ更新

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💡 次のステップ
# MAGIC
# MAGIC バッチ推論をマスターしたら、以下にチャレンジしてください：
# MAGIC
# MAGIC 1. **Databricks Workflows**と統合
# MAGIC    - 定期的なバッチジョブとしてスケジュール実行
# MAGIC    - 依存関係のある複数ジョブの連鎖実行
# MAGIC
# MAGIC 2. **Delta Lake最適化**
# MAGIC    - `OPTIMIZE`コマンドでファイル圧縮
# MAGIC    - `Z-ORDER`でクエリパフォーマンス向上
# MAGIC
# MAGIC 3. **MLflowとの統合**
# MAGIC    - バッチ推論結果をMLflowで追跡
# MAGIC    - モデルバージョン管理と実験比較
# MAGIC
# MAGIC 4. **Unity Catalogでのガバナンス**
# MAGIC    - データアクセス権限の管理
# MAGIC    - リネージ追跡でデータフローを可視化

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎓 まとめ
# MAGIC
# MAGIC Exercise 2では以下を学びました：
# MAGIC
# MAGIC ### メインコンテンツ
# MAGIC - Pydanticによる構造化スキーマ定義
# MAGIC - Structured Outputsでの情報抽出
# MAGIC - ビジネス分析のためのDataFrame変換
# MAGIC
# MAGIC ### おまけ: バッチ推論
# MAGIC - `ai_query()`による大規模データ処理
# MAGIC - Structured Streamingによる準リアルタイム処理
# MAGIC
# MAGIC これらのテクニックを組み合わせることで、
# MAGIC **プロトタイプから本番環境までスケールするLLMアプリケーション**を構築できます！

# COMMAND ----------


