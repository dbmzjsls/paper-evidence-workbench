import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import Config
from src.rag_chain import get_retriever_only, get_llm_only
from rag_eval.prompts import format_generation_prompt


def _build_eval_llm():
    """构建 RAGAS 评判用 LLM """
    return LangchainLLMWrapper(
        ChatOpenAI(
            api_key=Config.get_api_key(),
            base_url=Config.DASHSCOPE_BASE_URL,
            model=Config.LLM_EVAL,
            temperature=0, # 保障评估一致性
        )
    )


def _build_eval_embeddings():
    """构建 RAGAS 评判用 Embeddings"""
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
        )
    )


def evaluate_with_ragas(
    mode: str = "ensemble",
    test_data: list[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    对单一检索模式执行完整 RAGAS 评估。
    流程: 检索 → 生成 → 构建 SingleTurnSample → 计算 4 项指标
    """
    if test_data is None:
        from rag_eval.test_dataset import get_test_dataset
        test_data = get_test_dataset()

    retriever = get_retriever_only(mode=mode)
    llm = get_llm_only()
    eval_llm = _build_eval_llm()
    eval_embeddings = _build_eval_embeddings()

    metrics_map = {
        "faithfulness":       Faithfulness(llm=eval_llm),
        "answer_relevancy":   AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        "context_precision":  ContextPrecision(llm=eval_llm),
        "context_recall":     ContextRecall(llm=eval_llm),
    }

    samples = []
    for i, item in enumerate(test_data):
        q = item["question"]
        if verbose:
            print(f"  [{i+1}/{len(test_data)}] {q[:60]}...")

        try:
            # 第1步: 检索 — 获取真实 contexts
            retrieved_docs = retriever.invoke(q)
            contexts = [doc.page_content for doc in retrieved_docs]

            # 第2步: 生成 — 基于检索到的 contexts
            prompt = format_generation_prompt(q, contexts)
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            # 第3步: 构建 RAGAS 样本 — contexts 来自检索器
            sample = SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=item.get("ground_truth", ""),
            )
            samples.append(sample)
        except Exception as e:
            if verbose:
                print(f"    [SKIP] {e}")

    if not samples:
        return {"mode": mode, "error": "无有效样本"}

    # 第4步: 逐指标计算得分
    results = {"mode": mode, "metrics": {}, "samples_detail": [], "n_samples": len(samples)}

    metric_scores_by_sample = [dict() for _ in samples]

    for metric_name, metric_fn in metrics_map.items():
        scores = []
        for sample_index, sample in enumerate(samples):
            try:
                score = float(metric_fn.single_turn_score(sample))
                scores.append(score)
                metric_scores_by_sample[sample_index][metric_name] = score
            except Exception:
                scores.append(0.0)
                metric_scores_by_sample[sample_index][metric_name] = None
        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        results["metrics"][metric_name] = avg
        if verbose:
            print(f"  {metric_name}: {avg:.4f}")

    # 逐样本明细
    for i, sample in enumerate(samples):
        detail = {
            "question": sample.user_input[:80],
            "n_contexts": len(sample.retrieved_contexts),
            "scores": metric_scores_by_sample[i],
        }
        results["samples_detail"].append(detail)

    return results


def run_ablation_study(
    test_data: list[dict] = None,
    modes: list[str] = None,
    save: bool = True,
) -> dict:
    """
    执行消融实验：对比多种检索模式的 RAGAS 指标。

    Args:
        test_data: 测试数据
        modes: 要对比的模式，默认 ["vector", "ensemble", "rerank", "multiquery"]
        save: 是否保存到 experiments/

    Returns:
        {"vector": {...}, "ensemble": {...}, "rerank": {...}, "multiquery": {...}}
    """
    if test_data is None:
        from rag_eval.test_dataset import get_test_dataset
        test_data = get_test_dataset()

    if modes is None:
        modes = ["vector", "ensemble", "rerank", "multiquery"]

    print(f"\n{'='*70}")
    print(f"RAGAS 消融实验 — {len(test_data)} 条测试题, {len(modes)} 种检索模式")
    print(f"{'='*70}")

    all_results = {}
    for mode in modes:
        print(f"\n── 模式: {mode} ──")
        all_results[mode] = evaluate_with_ragas(mode=mode, test_data=test_data, verbose=True)

    _print_table(all_results)

    if save:
        os.makedirs("experiments", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"experiments/ablation_{ts}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {filepath}")

    return all_results


def _print_table(all_results: dict):
    """打印多模式对比表"""
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_cn = {
        "faithfulness":       "忠实度 (Faithfulness)",
        "answer_relevancy":   "答案相关性 (AnswerRelevancy)",
        "context_precision":  "上下文精确度 (ContextPrecision)",
        "context_recall":     "上下文召回率 (ContextRecall)",
    }
    modes = list(all_results.keys())

    print(f"\n{'='*90}")
    print(f"{'指标':<32}", end="")
    for m in modes:
        print(f" {m:<16}", end="")
    print()
    print("-" * 90)

    for mk in metric_keys:
        best = max(all_results[m].get("metrics", {}).get(mk, 0) for m in modes)
        print(f"{metric_cn.get(mk, mk):<32}", end="")
        for m in modes:
            v = all_results[m].get("metrics", {}).get(mk, 0)
            star = " *" if v == best and v > 0 else ""
            print(f" {v:<14.4f}{star}", end="")
        print()
    print("=" * 90)
    print("  * = 该指标最优\n")


if __name__ == "__main__":
    run_ablation_study()
