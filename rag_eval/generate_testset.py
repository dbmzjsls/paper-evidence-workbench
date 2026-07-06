import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.rag_chain import get_llm_only

def generate_qa_from_chunks(n_questions=15):
    """从文档库中随机抽取 chunk，让 LLM 生成提问和简答"""
    with open(Config.DOCS_DIR, "rb") as f:
        docs = pickle.load(f)

    # 随机抽样
    sample_docs = random.sample(docs, min(n_questions * 3, len(docs)))

    llm = get_llm_only()
    qa_pairs = []

    for i, doc in enumerate(sample_docs):
        prompt = f"""阅读以下学术论文片段，生成一个可以用该片段内容回答的具体问题。
要求:
1. 问题必须聚焦片段中的具体信息(如研究方法、样本量、发现、结论等)
2. 给出简短的事实型答案(不超过150字)
3. 输出格式: Q: 问题\nA: 答案

论文片段:
{doc.page_content[:1500]}

输出:"""

        try:
            result = llm.invoke(prompt).content
            lines = result.strip().split("\n")
            q = ""
            a = ""
            for line in lines:
                if line.startswith("Q:") or line.startswith("Q："):
                    q = line[2:].strip()
                elif line.startswith("A:") or line.startswith("A："):
                    a = line[2:].strip()
            if q and a:
                qa_pairs.append({"question": q, "ground_truth": a})
                print(f"  [{len(qa_pairs)}] {q[:60]}...")
        except Exception as e:
            print(f"  [SKIP] {e}")

        if len(qa_pairs) >= n_questions:
            break

    # 保存到文件
    out_path = "rag_eval/generated_qa.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n已生成 {len(qa_pairs)} 条QA，保存到 {out_path}")
    print("请人工检查并筛选后，手动添加到 test_dataset.py 中。")

if __name__ == "__main__":
    generate_qa_from_chunks(15)
