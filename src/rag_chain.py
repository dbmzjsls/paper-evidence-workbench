"""
Compatibility facade for the Paper Evidence Workbench retrieval service.

New code should use src.retrieval.ResearchRAG directly.
"""

from __future__ import annotations

from src.config import Config
from src.retrieval import ResearchRAG


class RAGChainAdapter:
    def __init__(self, mode: str = "ensemble"):
        self.mode = mode
        self.service = ResearchRAG()

    def invoke(self, question: str) -> str:
        return self.service.answer(question, mode=self.mode).answer

    async def astream(self, question: str):
        result = self.service.answer(question, mode=self.mode).answer
        yield result


class RetrieverAdapter:
    def __init__(self, mode: str = "ensemble"):
        self.mode = mode
        self.service = ResearchRAG()

    def invoke(self, question: str):
        from langchain_core.documents import Document

        results = self.service.retrieve(
            question,
            k=Config.RETRIEVAL_CONTEXTS,
            mode=self.mode,
        )
        return [
            Document(
                page_content=item.chunk.text,
                metadata={
                    "chunk_id": item.chunk.chunk_id,
                    "document_id": item.chunk.document_id,
                    "score": item.score,
                    **item.chunk.metadata,
                },
            )
            for item in results
        ]


def get_rag_chain(mode: str = "ensemble"):
    return RAGChainAdapter(mode=mode)


def get_retriever_only(mode: str = "ensemble"):
    return RetrieverAdapter(mode=mode)


def get_llm_only():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=Config.get_api_key(),
        base_url=Config.DASHSCOPE_BASE_URL,
        model=Config.LLM_MODEL,
        temperature=0,
    )
