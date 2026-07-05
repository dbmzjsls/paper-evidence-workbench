"""
RAG 评估 — 基础检索 vs 增强检索对比

对比实验:
- 基础检索 (ensemble) : BM25 + FAISS 混合检索
- 增强检索 (multiquery): BM25 + FAISS + LLM 多查询扩展

使用共享评估模块 evaluator.py

用法:
    uv run python -m rag_eval.eval_main
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.evaluator import run_ablation_study
from rag_eval.test_dataset import get_test_dataset


def main():
    test_data = get_test_dataset()
    print(f"测试集: {len(test_data)} 条\n")
    run_ablation_study(test_data=test_data, modes=["ensemble", "multiquery"], save=True)


if __name__ == "__main__":
    main()
