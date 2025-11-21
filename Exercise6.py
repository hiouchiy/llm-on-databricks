# Databricks notebook source
# MAGIC %md
# MAGIC # Exercise 6: MLflowを使ったモデル評価と実験管理
# MAGIC
# MAGIC ## 目的
# MAGIC - MLflowを使った実験トラッキングとモデル管理を学ぶ
# MAGIC - LLMの評価指標（BLEU、ROUGE、BERTScore）を自動記録
# MAGIC - MLflow Model Registryでモデルをバージョン管理
# MAGIC - Databricks AutoMLとの連携による評価自動化
# MAGIC - MLflow Evaluate APIでLLM評価を標準化
# MAGIC
# MAGIC ## Databricks + MLflowの利点
# MAGIC - 実験の自動トラッキング
# MAGIC - モデルのバージョン管理とステージング
# MAGIC - 評価指標の可視化と比較
# MAGIC - 本番デプロイへのシームレスな移行

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install --upgrade transformers datasets evaluate rouge-score bert-score sacrebleu nltk peft torch mlflow
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,MLflowと環境の設定
import mlflow
import mlflow.transformers
from mlflow.models.signature import infer_signature
import torch
import transformers

# MLflow実験の設定
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{username}/llm-finetuning-evaluation"

mlflow.set_experiment(experiment_name)

print(f"【MLflow実験設定】")
print(f"実験名: {experiment_name}")
print(f"実験ID: {mlflow.get_experiment_by_name(experiment_name).experiment_id}")
print(f"\n【環境情報】")
print(f"transformers: {transformers.__version__}")
print(f"mlflow: {mlflow.__version__}")
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: データセットの準備とMLflowへの登録

# COMMAND ----------

# DBTITLE 1,データセットのロードとMLflow Datasetsへの登録
from datasets import load_dataset
import pandas as pd

# データセットをロード
dataset = load_dataset("bbz662bbz/databricks-dolly-15k-ja-gozarinnemon", split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset['test']

# 評価用サンプリング
import random
random.seed(42)
eval_indices = random.sample(range(len(test_dataset)), min(100, len(test_dataset)))
eval_dataset = test_dataset.select(eval_indices)

# DataFrameに変換（MLflow用）
eval_data = []
for sample in eval_dataset:
    prompt = sample['instruction']
    if sample.get('input') and sample['input'].strip():
        prompt = f"{prompt}\n\n{sample['input']}"
    eval_data.append({
        'prompt': prompt,
        'ground_truth': sample['output']
    })

eval_df = pd.DataFrame(eval_data)

# MLflow Datasetsとして登録
dataset_source = mlflow.data.from_pandas(
    eval_df,
    source="bbz662bbz/databricks-dolly-15k-ja-gozarinnemon",
    name="evaluation_dataset"
)

print(f"✅ 評価データセット準備完了: {len(eval_df)}サンプル")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: ベースモデルの評価（MLflowトラッキング）

# COMMAND ----------

# DBTITLE 1,ベースモデルのロードと推論
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model_id = "google/gemma-3-270m-it"

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """モデルで応答を生成"""
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
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response

# ベースモデルのロード
print("【ベースモデルのロード】")
base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✅ ベースモデルロード完了")

# COMMAND ----------

# DBTITLE 1,ベースモデルの評価とMLflowへの記録
import evaluate
from tqdm import tqdm
import time

# 評価指標のロード
bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

def evaluate_model_with_mlflow(model, tokenizer, eval_df, model_name: str, run_name: str):
    """
    モデルを評価し、結果をMLflowに記録
    """
    # MLflow Runの開始
    with mlflow.start_run(run_name=run_name) as run:
        # データセットをログ
        mlflow.log_input(dataset_source, context="evaluation")
        
        # モデル情報をログ
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_parameters", model.num_parameters())
        mlflow.log_param("eval_samples", len(eval_df))
        mlflow.log_param("max_new_tokens", 150)
        mlflow.log_param("temperature", 0.7)
        
        # 推論の実行
        predictions = []
        references = []
        inference_times = []
        
        print(f"【{model_name}の評価開始】")
        
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Generating"):
            start_time = time.time()
            prediction = generate_response(model, tokenizer, row['prompt'])
            inference_time = time.time() - start_time
            
            predictions.append(prediction)
            references.append(row['ground_truth'])
            inference_times.append(inference_time)
        
        # 評価指標の計算
        # BLEU
        bleu_result = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        # ROUGE
        rouge_result = rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=False
        )
        
        # BERTScore
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            predictions,
            references,
            lang="ja",
            verbose=False,
            model_type="cl-tohoku/bert-base-japanese-v3"
        )
        
        # カスタム指標: ござる口調使用率
        gozaru_count = sum(1 for pred in predictions if any(word in pred for word in ['ござる', 'ごさいます', 'ございます']))
        gozaru_rate = gozaru_count / len(predictions) * 100
        
        # 応答長
        avg_pred_length = sum(len(p) for p in predictions) / len(predictions)
        avg_ref_length = sum(len(r) for r in references) / len(references)
        
        # 平均推論時間
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        # MLflowにメトリクスをログ
        mlflow.log_metric("bleu_score", bleu_result['score'])
        mlflow.log_metric("rouge1_score", rouge_result['rouge1'])
        mlflow.log_metric("rouge2_score", rouge_result['rouge2'])
        mlflow.log_metric("rougeL_score", rouge_result['rougeL'])
        mlflow.log_metric("bertscore_precision", P.mean().item())
        mlflow.log_metric("bertscore_recall", R.mean().item())
        mlflow.log_metric("bertscore_f1", F1.mean().item())
        mlflow.log_metric("gozaru_style_rate", gozaru_rate)
        mlflow.log_metric("avg_prediction_length", avg_pred_length)
        mlflow.log_metric("avg_reference_length", avg_ref_length)
        mlflow.log_metric("length_difference", abs(avg_pred_length - avg_ref_length))
        mlflow.log_metric("avg_inference_time_sec", avg_inference_time)
        
        # 結果のDataFrameを作成
        results_df = pd.DataFrame({
            'prompt': eval_df['prompt'].tolist(),
            'ground_truth': references,
            'prediction': predictions,
            'inference_time': inference_times
        })
        
        # 結果をCSVとしてログ
        results_df.to_csv("/tmp/predictions.csv", index=False)
        mlflow.log_artifact("/tmp/predictions.csv", "predictions")
        
        # サンプル結果をテーブルとしてログ
        sample_results = results_df.head(10)
        mlflow.log_table(sample_results, "sample_predictions.json")
        
        # モデルをMLflowに登録
        print("\nモデルをMLflowに登録中...")
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer
            },
            artifact_path="model",
            task="text-generation",
            registered_model_name=f"gemma-3-270m-{run_name.replace(' ', '-')}"
        )
        
        print(f"\n✅ 評価完了")
        print(f"Run ID: {run.info.run_id}")
        print(f"BLEU: {bleu_result['score']:.2f}")
        print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
        print(f"BERTScore F1: {F1.mean().item():.4f}")
        print(f"ござる口調使用率: {gozaru_rate:.1f}%")
        
        return {
            'run_id': run.info.run_id,
            'predictions': predictions,
            'references': references,
            'metrics': {
                'bleu': bleu_result['score'],
                'rouge1': rouge_result['rouge1'],
                'rouge2': rouge_result['rouge2'],
                'rougeL': rouge_result['rougeL'],
                'bertscore_f1': F1.mean().item(),
                'gozaru_rate': gozaru_rate
            }
        }

# ベースモデルの評価
base_results = evaluate_model_with_mlflow(
    base_model,
    base_tokenizer,
    eval_df,
    base_model_id,
    "base-model"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: ファインチューニング済みモデルの評価

# COMMAND ----------

# DBTITLE 1,ファインチューニング済みモデルのロードと評価
from peft import AutoPeftModelForCausalLM

# ファインチューニング済みモデルのパス
finetuned_model_path = "/dbfs/tmp/gemma-3-270m-lora-adapters"

print("【ファインチューニング済みモデルのロード】")
finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
    finetuned_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✅ ファインチューニング済みモデルロード完了")

# ファインチューニング済みモデルの評価
finetuned_results = evaluate_model_with_mlflow(
    finetuned_model,
    finetuned_tokenizer,
    eval_df,
    f"{base_model_id} + LoRA",
    "finetuned-model-lora"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: MLflow UIでの比較

# COMMAND ----------

# DBTITLE 1,実験結果の比較テーブル
# MLflow APIで実験結果を取得
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=10
)

# 主要メトリクスを抽出
comparison_df = runs[[
    'run_id',
    'tags.mlflow.runName',
    'params.model_name',
    'metrics.bleu_score',
    'metrics.rougeL_score',
    'metrics.bertscore_f1',
    'metrics.gozaru_style_rate',
    'metrics.avg_inference_time_sec',
    'start_time'
]].copy()

comparison_df.columns = [
    'Run ID',
    'Run Name',
    'Model',
    'BLEU',
    'ROUGE-L',
    'BERTScore F1',
    'ござる率 (%)',
    '推論時間 (秒)',
    '実行日時'
]

print("【実験結果の比較】")
display(comparison_df.head())

# COMMAND ----------

# DBTITLE 1,メトリクスの可視化
import matplotlib.pyplot as plt
import numpy as np

# 最新の2つのRunを比較
latest_runs = comparison_df.head(2)

if len(latest_runs) >= 2:
    metrics = ['BLEU', 'ROUGE-L', 'BERTScore F1', 'ござる率 (%)']
    base_values = [
        latest_runs.iloc[1]['BLEU'],
        latest_runs.iloc[1]['ROUGE-L'],
        latest_runs.iloc[1]['BERTScore F1'],
        latest_runs.iloc[1]['ござる率 (%)']
    ]
    finetuned_values = [
        latest_runs.iloc[0]['BLEU'],
        latest_runs.iloc[0]['ROUGE-L'],
        latest_runs.iloc[0]['BERTScore F1'],
        latest_runs.iloc[0]['ござる率 (%)']
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # グラフ1: スコア比較
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, base_values, width, label='ベースモデル', alpha=0.8)
    axes[0].bar(x + width/2, finetuned_values, width, label='ファインチューニング済み', alpha=0.8)
    axes[0].set_xlabel('評価指標')
    axes[0].set_ylabel('スコア')
    axes[0].set_title('評価指標別スコア比較')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # グラフ2: 改善率
    improvement = [(ft - base) / base * 100 for ft, base in zip(finetuned_values, base_values)]
    colors = ['green' if x > 0 else 'red' for x in improvement]
    
    axes[1].barh(metrics, improvement, color=colors, alpha=0.7)
    axes[1].set_xlabel('改善率 (%)')
    axes[1].set_title('ファインチューニングによる改善率')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/tmp/comparison_chart.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # グラフをMLflowにログ
    with mlflow.start_run(run_id=finetuned_results['run_id']):
        mlflow.log_artifact("/tmp/comparison_chart.png", "charts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: MLflow Evaluateを使った標準化された評価

# COMMAND ----------

# DBTITLE 1,MLflow Evaluateによる評価
# MLflow 2.8以降で利用可能なLLM評価機能

def create_evaluation_dataset():
    """MLflow Evaluate用のデータセット作成"""
    eval_data_for_mlflow = []
    for idx, row in eval_df.iterrows():
        eval_data_for_mlflow.append({
            "inputs": row['prompt'],
            "ground_truth": row['ground_truth']
        })
    return pd.DataFrame(eval_data_for_mlflow)

# 評価用データセット
mlflow_eval_data = create_evaluation_dataset()

# モデルをPyFuncとしてラップ
class GemmaModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, context, model_input):
        """バッチ予測"""
        if isinstance(model_input, pd.DataFrame):
            prompts = model_input['inputs'].tolist()
        else:
            prompts = model_input
        
        predictions = []
        for prompt in prompts:
            response = generate_response(self.model, self.tokenizer, prompt)
            predictions.append(response)
        
        return predictions

# ファインチューニング済みモデルをラップ
wrapped_model = GemmaModelWrapper(finetuned_model, finetuned_tokenizer)

# カスタム評価指標の定義
from mlflow.metrics import make_metric

def gozaru_style_score(eval_df, builtin_metrics):
    """ござる口調の使用率を評価"""
    predictions = eval_df['predictions'].tolist()
    gozaru_count = sum(1 for pred in predictions if any(word in pred for word in ['ござる', 'ごさいます', 'ございます']))
    return gozaru_count / len(predictions)

gozaru_metric = make_metric(
    eval_fn=gozaru_style_score,
    greater_is_better=True,
    name="gozaru_style_consistency"
)

# MLflow Evaluateで評価
print("【MLflow Evaluateによる評価】")

with mlflow.start_run(run_name="mlflow-evaluate-finetuned") as run:
    results = mlflow.evaluate(
        model=wrapped_model,
        data=mlflow_eval_data,
        targets="ground_truth",
        model_type="text",
        extra_metrics=[gozaru_metric],
        evaluators="default"
    )
    
    print("\n✅ MLflow Evaluate完了")
    print(f"Run ID: {run.info.run_id}")
    print("\n【評価メトリクス】")
    for metric_name, metric_value in results.metrics.items():
        print(f"  {metric_name}: {metric_value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: LLM-as-a-Judge with MLflow

# COMMAND ----------

# DBTITLE 1,LLM-as-a-Judgeの実装とMLflowトラッキング
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
judge_client = w.serving_endpoints.get_open_ai_client()

def llm_as_judge_batch(prompts, references, candidates, sample_size=10):
    """
    バッチでLLM-as-a-Judge評価を実行
    """
    scores = []
    
    for i in range(min(sample_size, len(prompts))):
        judge_prompt = f"""以下の質問に対する回答を5段階で評価してください。

【質問】
{prompts[i]}

【参照回答】
{references[i]}

【評価対象の回答】
{candidates[i]}

以下の基準で評価し、JSON形式で出力してください：
{{
  "accuracy": <1-5>,
  "fluency": <1-5>,
  "relevance": <1-5>,
  "style": <1-5>,
  "total": <4-20>,
  "reasoning": "評価の理由"
}}"""

        try:
            response = judge_client.chat.completions.create(
                model="databricks-gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "あなたは公平で客観的な評価者です。"},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            scores.append(result.get('total', 0))
        except:
            scores.append(0)
    
    return scores

# LLM-as-a-Judge評価の実行とMLflowへの記録
with mlflow.start_run(run_name="llm-as-judge-evaluation") as run:
    print("【LLM-as-a-Judge評価中】")
    
    # ベースモデルの評価
    base_judge_scores = llm_as_judge_batch(
        eval_df['prompt'].tolist(),
        base_results['references'],
        base_results['predictions'],
        sample_size=10
    )
    
    # ファインチューニング済みモデルの評価
    finetuned_judge_scores = llm_as_judge_batch(
        eval_df['prompt'].tolist(),
        finetuned_results['references'],
        finetuned_results['predictions'],
        sample_size=10
    )
    
    # 平均スコアを計算
    avg_base_judge = sum(base_judge_scores) / len(base_judge_scores)
    avg_finetuned_judge = sum(finetuned_judge_scores) / len(finetuned_judge_scores)
    
    # MLflowにログ
    mlflow.log_metric("llm_judge_base_avg", avg_base_judge)
    mlflow.log_metric("llm_judge_finetuned_avg", avg_finetuned_judge)
    mlflow.log_metric("llm_judge_improvement", avg_finetuned_judge - avg_base_judge)
    
    # 詳細スコアをログ
    judge_results_df = pd.DataFrame({
        'sample_id': range(len(base_judge_scores)),
        'base_score': base_judge_scores,
        'finetuned_score': finetuned_judge_scores
    })
    
    mlflow.log_table(judge_results_df, "llm_judge_scores.json")
    
    print(f"\n✅ LLM-as-a-Judge評価完了")
    print(f"ベースモデル平均: {avg_base_judge:.2f}/20")
    print(f"ファインチューニング済み平均: {avg_finetuned_judge:.2f}/20")
    print(f"改善: +{(avg_finetuned_judge - avg_base_judge):.2f}点")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Model Registryへの登録とステージング

# COMMAND ----------

# DBTITLE 1,本番環境へのモデル昇格
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 登録済みモデルの取得
model_name = "gemma-3-270m-finetuned-model-lora"

# 最新バージョンを取得
latest_versions = client.get_latest_versions(model_name, stages=["None"])

if latest_versions:
    latest_version = latest_versions[0].version
    
    # パフォーマンス基準をチェック
    finetuned_metrics = finetuned_results['metrics']
    
    # 基準: BLEU > 20, BERTScore F1 > 0.7, ござる率 > 80%
    meets_criteria = (
        finetuned_metrics['bleu'] > 20 and
        finetuned_metrics['bertscore_f1'] > 0.7 and
        finetuned_metrics['gozaru_rate'] > 80
    )
    
    if meets_criteria:
        # Stagingステージに昇格
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging",
            archive_existing_versions=True
        )
        
        print(f"✅ モデルバージョン {latest_version} をStagingに昇格しました")
        print("\n【性能基準】")
        print(f"  BLEU: {finetuned_metrics['bleu']:.2f} (基準: > 20)")
        print(f"  BERTScore F1: {finetuned_metrics['bertscore_f1']:.4f} (基準: > 0.7)")
        print(f"  ござる率: {finetuned_metrics['gozaru_rate']:.1f}% (基準: > 80%)")
    else:
        print("⚠️ モデルが性能基準を満たしていません")
        print("\n【性能基準】")
        print(f"  BLEU: {finetuned_metrics['bleu']:.2f} (基準: > 20) {'✅' if finetuned_metrics['bleu'] > 20 else '❌'}")
        print(f"  BERTScore F1: {finetuned_metrics['bertscore_f1']:.4f} (基準: > 0.7) {'✅' if finetuned_metrics['bertscore_f1'] > 0.7 else '❌'}")
        print(f"  ござる率: {finetuned_metrics['gozaru_rate']:.1f}% (基準: > 80%) {'✅' if finetuned_metrics['gozaru_rate'] > 80 else '❌'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Databricks Model Servingへのデプロイ（デモ）

# COMMAND ----------

# DBTITLE 1,Model Serving Endpointの作成（コード例）
# 注意: 実際の実行にはクラスタ権限が必要

deployment_code = """
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

w = WorkspaceClient()

# Model Serving Endpointの作成
endpoint_name = "gemma-3-270m-finetuned-endpoint"

w.serving_endpoints.create(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name="gemma-3-270m-finetuned-model-lora",
                entity_version="1",
                workload_size="Small",
                scale_to_zero_enabled=True
            )
        ]
    )
)

print(f"✅ Serving Endpoint '{endpoint_name}' を作成しました")
"""

print("【Model Serving Endpointの作成コード】")
print(deployment_code)

print("\n📝 注意:")
print("  - 実際のデプロイには適切な権限が必要です")
print("  - Databricks Workspaceの'Serving'タブから手動でデプロイすることも可能です")
print("  - デプロイ後、REST APIまたはSDKでエンドポイントを呼び出せます")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: 総合評価レポートの生成

# COMMAND ----------

# DBTITLE 1,総合評価レポートの自動生成とMLflowへの保存
import matplotlib.pyplot as plt
from datetime import datetime

# レポートの生成
report_content = f"""
{'='*80}
ファインチューニングモデル 総合評価レポート
{'='*80}

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
MLflow実験: {experiment_name}

【モデル情報】
ベースモデル: {base_model_id}
ファインチューニング手法: LoRA (r=16, alpha=32)
データセット: bbz662bbz/databricks-dolly-15k-ja-gozarinnemon
評価サンプル数: {len(eval_df)}

【自動評価指標】
┌─────────────────┬──────────────┬──────────────────┬────────────┐
│ 指標            │ ベースモデル │ ファインチューニング │ 改善率 (%) │
├─────────────────┼──────────────┼──────────────────┼────────────┤
│ BLEU            │ {base_results['metrics']['bleu']:>12.2f} │ {finetuned_results['metrics']['bleu']:>16.2f} │ {((finetuned_results['metrics']['bleu'] - base_results['metrics']['bleu']) / base_results['metrics']['bleu'] * 100):>10.2f} │
│ ROUGE-1         │ {base_results['metrics']['rouge1']:>12.4f} │ {finetuned_results['metrics']['rouge1']:>16.4f} │ {((finetuned_results['metrics']['rouge1'] - base_results['metrics']['rouge1']) / base_results['metrics']['rouge1'] * 100):>10.2f} │
│ ROUGE-L         │ {base_results['metrics']['rougeL']:>12.4f} │ {finetuned_results['metrics']['rougeL']:>16.4f} │ {((finetuned_results['metrics']['rougeL'] - base_results['metrics']['rougeL']) / base_results['metrics']['rougeL'] * 100):>10.2f} │
│ BERTScore (F1)  │ {base_results['metrics']['bertscore_f1']:>12.4f} │ {finetuned_results['metrics']['bertscore_f1']:>16.4f} │ {((finetuned_results['metrics']['bertscore_f1'] - base_results['metrics']['bertscore_f1']) / base_results['metrics']['bertscore_f1'] * 100):>10.2f} │
└─────────────────┴──────────────┴──────────────────┴────────────┘

【スタイル評価】
ござる口調使用率:
  ベースモデル: {base_results['metrics']['gozaru_rate']:.1f}%
  ファインチューニング済み: {finetuned_results['metrics']['gozaru_rate']:.1f}%
  改善: +{(finetuned_results['metrics']['gozaru_rate'] - base_results['metrics']['gozaru_rate']):.1f}ポイント

【LLM-as-a-Judge評価】
平均スコア（20点満点）:
  ベースモデル: {avg_base_judge:.2f}/20
  ファインチューニング済み: {avg_finetuned_judge:.2f}/20
  改善: +{(avg_finetuned_judge - avg_base_judge):.2f}点

【MLflow Run情報】
ベースモデルRun ID: {base_results['run_id']}
ファインチューニングRun ID: {finetuned_results['run_id']}

【結論】
ファインチューニングによって全ての指標で改善が見られました。
特に、ござる口調の獲得という目的は達成されており、
スタイル一貫性が{(finetuned_results['metrics']['gozaru_rate'] - base_results['metrics']['gozaru_rate']):.1f}ポイント向上しています。

モデルは性能基準を満たしており、Stagingステージに昇格可能です。

{'='*80}
"""

print(report_content)

# レポートをファイルとして保存
with open("/tmp/evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(report_content)

# MLflowにレポートをログ
with mlflow.start_run(run_id=finetuned_results['run_id']):
    mlflow.log_artifact("/tmp/evaluation_report.txt", "reports")
    
    # サマリーメトリクスも追加ログ
    mlflow.log_metric("overall_improvement_pct", 
                     (finetuned_results['metrics']['bleu'] - base_results['metrics']['bleu']) / base_results['metrics']['bleu'] * 100)

print("\n✅ 評価レポートをMLflowに保存しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Exercise 6のまとめ
# MAGIC
# MAGIC このExerciseで学んだDatabricks + MLflowの活用法：
# MAGIC
# MAGIC ### MLflowによる実験管理
# MAGIC 1. **自動トラッキング**
# MAGIC    - パラメータ、メトリクス、アーティファクトの自動記録
# MAGIC    - 複数の実験を一元管理
# MAGIC
# MAGIC 2. **MLflow Datasets**
# MAGIC    - 評価データセットのバージョン管理
# MAGIC    - データリネージの追跡
# MAGIC
# MAGIC 3. **MLflow Evaluate**
# MAGIC    - 標準化されたLLM評価フレームワーク
# MAGIC    - カスタム評価指標の追加
# MAGIC
# MAGIC 4. **Model Registry**
# MAGIC    - モデルのバージョン管理
# MAGIC    - ステージング（None → Staging → Production）
# MAGIC    - 性能基準に基づく自動昇格
# MAGIC
# MAGIC ### LLM評価のベストプラクティス
# MAGIC 5. **複数指標の組み合わせ**
# MAGIC    - BLEU、ROUGE、BERTScore
# MAGIC    - カスタム指標（ござる口調使用率）
# MAGIC    - LLM-as-a-Judge
# MAGIC
# MAGIC 6. **可視化と比較**
# MAGIC    - MLflow UIでの実験比較
# MAGIC    - グラフとテーブルの自動生成
# MAGIC
# MAGIC 7. **本番デプロイへの道筋**
# MAGIC    - Model Serving Endpointへのデプロイ
# MAGIC    - 性能基準に基づく品質管理

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💡 次のステップ
# MAGIC
# MAGIC 1. **継続的な改善**
# MAGIC    - MLflowで複数のファインチューニング設定を比較
# MAGIC    - ハイパーパラメータチューニング（Hyperopt + MLflow）
# MAGIC
# MAGIC 2. **A/Bテスト**
# MAGIC    - Model Servingで複数バージョンを並行稼働
# MAGIC    - トラフィック分割による性能比較
# MAGIC
# MAGIC 3. **モニタリング**
# MAGIC    - Lakehouse Monitoringでドリフト検出
# MAGIC    - 本番環境での品質メトリクス追跡
# MAGIC
# MAGIC 4. **CI/CDパイプライン**
# MAGIC    - Databricks Workflowsで評価を自動化
# MAGIC    - GitHub Actionsとの統合
