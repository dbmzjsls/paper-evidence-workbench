"""
RAG 评估入口

用法:
    uv run python -m rag_eval.eval                      # 全部 4 组消融实验
    uv run python -m rag_eval.eval --mode rerank         # 单模式评估
    uv run python -m rag_eval.eval --modes vector,ensemble  # 指定对比模式
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.evaluator import evaluate_with_ragas, run_ablation_study
from rag_eval.test_dataset import get_test_dataset


def main():
    parser = argparse.ArgumentParser(description="RAGAS RAG 评估")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["vector", "ensemble", "rerank", "multiquery"],
        help="单模式评估",
    )
    parser.add_argument(
        "--modes", type=str, default=None,
        help="逗号分隔的模式列表，如 'vector,ensemble,rerank'",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="不保存结果到文件",
    )
    args = parser.parse_args()

    test_data = get_test_dataset()
    print(f"测试集: {len(test_data)} 条\n")

    if args.mode:
        result = evaluate_with_ragas(mode=args.mode, test_data=test_data, verbose=True)
        print(f"\n评估完成: {args.mode}")
        for k, v in result.get("metrics", {}).items():
            print(f"  {k}: {v:.4f}")
    elif args.modes:
        mode_list = [m.strip() for m in args.modes.split(",")]
        run_ablation_study(test_data=test_data, modes=mode_list, save=not args.no_save)
    else:
        run_ablation_study(test_data=test_data, save=not args.no_save)


if __name__ == "__main__":
    main()
