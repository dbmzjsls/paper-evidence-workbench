from __future__ import annotations


def format_generation_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(contexts)
    return (
        "请根据以下背景信息回答用户问题。"
        "如果背景信息中无法得出答案，请明确说明。\n\n"
        f"背景信息：\n{context_text}\n\n"
        f"用户问题：{question}\n\n"
        "回答："
    )
