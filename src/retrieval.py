from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.models import (
    Citation,
    EvidenceChunk,
    QueryResult,
    ScreeningItem,
    ScreeningReport,
)
from src.storage import CorpusStorage


@dataclass
class RetrievedEvidence:
    chunk: EvidenceChunk
    score: float


class ResearchRAG:
    def __init__(self, storage: CorpusStorage | None = None):
        self.storage = storage or CorpusStorage()
        self._vectorstore = None
        self._reranker = None
        self._reranker_unavailable = False

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        mode: str = "ensemble",
        include_keywords: list[str] | None = None,
        exclude_keywords: list[str] | None = None,
    ) -> list[RetrievedEvidence]:
        k = k or Config.RETRIEVAL_CONTEXTS
        mode = self._normalize_mode(mode)
        include_keywords = [kw.strip() for kw in include_keywords or [] if kw.strip()]
        exclude_keywords = [kw.strip() for kw in exclude_keywords or [] if kw.strip()]
        queries = self._queries_for_mode(query, mode)

        candidate_limit = max(k * 20, Config.KEYWORD_CANDIDATE_LIMIT)
        keyword_candidates: list[EvidenceChunk] = []
        if mode != "vector":
            for search_query in queries:
                keyword_candidates.extend(
                    self.storage.search_chunks(
                        search_query,
                        limit=candidate_limit,
                        include_keywords=include_keywords,
                        exclude_keywords=exclude_keywords,
                    )
                )
            keyword_candidates = list(
                {chunk.chunk_id: chunk for chunk in keyword_candidates}.values()
            )
        by_id = {chunk.chunk_id: chunk for chunk in keyword_candidates}
        scores: dict[str, float] = {}

        if mode in {"vector", "ensemble", "rerank", "multiquery"}:
            for search_query in queries:
                vector_hits = self._vector_rank_scores(
                    search_query,
                    k=max(k * 4, Config.VECTOR_K),
                )
                for chunk in self.storage.get_chunks_by_ids(vector_hits.keys()):
                    if self._passes_filters(chunk, include_keywords, exclude_keywords):
                        by_id[chunk.chunk_id] = chunk

                for chunk_id, score in vector_hits.items():
                    if chunk_id in by_id:
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + score * 0.65

        if mode != "vector":
            for search_query in queries:
                keyword_scores = self._keyword_scores(search_query, keyword_candidates)
                for chunk_id, score in keyword_scores.items():
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + score * 0.35

        if not scores:
            for chunk in by_id.values():
                scores[chunk.chunk_id] = self._keyword_score(query, chunk.search_text)
        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        candidate_count = max(k, min(len(ranked), Config.RERANK_CANDIDATE_LIMIT))
        candidates = [
            RetrievedEvidence(chunk=by_id[chunk_id], score=score)
            for chunk_id, score in ranked[:candidate_count]
            if chunk_id in by_id
        ]
        return self._rerank(query, candidates, force=mode == "rerank")[:k]

    def answer(self, question: str, mode: str = "ensemble") -> QueryResult:
        retrieved = self.retrieve(question, k=Config.RETRIEVAL_CONTEXTS, mode=mode)
        if not retrieved:
            return QueryResult(
                question=question,
                answer="当前语料库中没有可支持回答的证据。请先导入论文，或换一个更具体的研究问题。",
                citations=[],
                contexts=[],
            )

        citations = [self._citation(i, item.chunk) for i, item in enumerate(retrieved, 1)]
        chunks = [item.chunk for item in retrieved]
        answer = self._llm_answer(question, chunks, citations) or self._extractive_answer(
            question, citations
        )
        return QueryResult(question=question, answer=answer, citations=citations, contexts=chunks)

    def screen(
        self,
        topic: str,
        limit: int | None = None,
        include_keywords: list[str] | None = None,
        exclude_keywords: list[str] | None = None,
    ) -> ScreeningReport:
        limit = min(limit or Config.SCREEN_DEFAULT_LIMIT, Config.SCREEN_MAX_LIMIT)
        retrieved = self.retrieve(
            topic,
            k=max(limit * 8, Config.RETRIEVAL_CONTEXTS),
            mode="ensemble",
            include_keywords=include_keywords,
            exclude_keywords=exclude_keywords,
        )

        grouped: dict[str, list[RetrievedEvidence]] = {}
        for item in retrieved:
            grouped.setdefault(item.chunk.document_id, []).append(item)

        items: list[ScreeningItem] = []
        for document_id, evidences in grouped.items():
            document = self.storage.get_document(document_id)
            if not document:
                continue
            top = sorted(evidences, key=lambda e: e.score, reverse=True)[:4]
            citations = [self._citation(i, evidence.chunk) for i, evidence in enumerate(top, 1)]
            score = min(1.0, sum(e.score for e in top) / max(len(top), 1))
            combined = "\n".join(c.quote for c in citations)
            items.append(
                ScreeningItem(
                    document_id=document.document_id,
                    title=document.title,
                    filename=document.filename,
                    relevance_score=round(score, 4),
                    decision=self._decision(score),
                    core_contribution=self._extract_sentence(
                        combined,
                        ["contribution", "贡献", "提出", "propose", "framework", "模型"],
                    ),
                    methods_data=self._extract_sentence(
                        combined,
                        ["method", "方法", "data", "数据", "experiment", "实证", "模型"],
                    ),
                    main_findings=self._extract_sentence(
                        combined,
                        ["find", "结果", "发现", "表明", "show", "effect", "影响"],
                    ),
                    limitations=self._extract_sentence(
                        combined,
                        ["limit", "局限", "future", "不足", "挑战", "risk"],
                        default="检索证据中未发现明确局限，需要精读原文确认。",
                    ),
                    research_ideas=self._research_ideas(topic, document.title, combined),
                    citations=citations,
                )
            )

        items.sort(key=lambda item: item.relevance_score, reverse=True)
        report = ScreeningReport(
            report_id=f"report_{uuid.uuid4().hex[:12]}",
            topic=topic,
            items=items[:limit],
            assumptions=[
                "Scores are based on retrieved evidence, not a full peer-review judgment.",
                "Research ideas are marked as hypotheses and should be verified by reading the cited passages.",
            ],
        )
        self.storage.save_report(report)
        return report

    def _vector_rank_scores(self, query: str, k: int) -> dict[str, float]:
        index_path = Path(Config.VECTOR_DIR) / "index.faiss"
        if not index_path.exists():
            return {}
        try:
            vectorstore = self._load_vectorstore()
            docs = vectorstore.similarity_search(query, k=k)
        except Exception:
            return {}
        total = max(len(docs), 1)
        scores: dict[str, float] = {}
        for rank, doc in enumerate(docs, 1):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                scores[chunk_id] = (total - rank + 1) / total
        return scores

    def _load_vectorstore(self):
        if self._vectorstore is None:
            if not Config.TRUST_LOCAL_FAISS_INDEX:
                raise RuntimeError(
                    "Local FAISS loading is disabled. Set TRUST_LOCAL_FAISS_INDEX=true "
                    "only for trusted index files."
                )
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": Config.EMBEDDING_DEVICE},
            )
            self._vectorstore = FAISS.load_local(
                Config.VECTOR_DIR,
                embeddings=embeddings,
                allow_dangerous_deserialization=Config.TRUST_LOCAL_FAISS_INDEX,
            )
        return self._vectorstore

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        force: bool = False,
    ) -> list[RetrievedEvidence]:
        if len(candidates) < 2 or not (force or Config.ENABLE_RERANK) or not Config.RERANK_MODEL:
            return candidates
        try:
            rerank_scores = self._rerank_scores(
                query,
                [candidate.chunk.search_text or candidate.chunk.text for candidate in candidates],
            )
        except Exception:
            return candidates
        if len(rerank_scores) != len(candidates):
            return candidates

        rescored: list[RetrievedEvidence] = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            # Keep a small contribution from the hybrid score for deterministic tie-breaking.
            score = float(rerank_score) + (candidate.score * 0.001)
            rescored.append(RetrievedEvidence(chunk=candidate.chunk, score=score))
        return sorted(rescored, key=lambda item: item.score, reverse=True)

    def _normalize_mode(self, mode: str) -> str:
        if mode in {"vector", "ensemble", "rerank", "multiquery"}:
            return mode
        return "ensemble"

    def _queries_for_mode(self, query: str, mode: str) -> list[str]:
        if mode != "multiquery":
            return [query]
        variants = [query]
        tokens = self._tokens(query)
        if tokens:
            variants.append(" ".join(tokens[:8]))
        if len(tokens) > 3:
            variants.append(" ".join(reversed(tokens[:8])))
        return list(dict.fromkeys(item for item in variants if item.strip()))

    def _rerank_scores(self, query: str, passages: list[str]) -> list[float]:
        reranker = self._load_reranker()
        pairs = [[query, passage] for passage in passages]
        scores = reranker.predict(pairs)
        return [float(score) for score in scores]

    def _load_reranker(self):
        if self._reranker_unavailable:
            raise RuntimeError("Reranker is unavailable")
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    Config.RERANK_MODEL,
                    device=Config.EMBEDDING_DEVICE,
                )
            except Exception as exc:
                self._reranker_unavailable = True
                raise RuntimeError("Unable to load reranker") from exc
        return self._reranker

    def _keyword_scores(self, query: str, chunks: list[EvidenceChunk]) -> dict[str, float]:
        raw_scores = {
            chunk.chunk_id: self._keyword_score(query, chunk.search_text) for chunk in chunks
        }
        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 0:
            return {}
        return {chunk_id: score / max_score for chunk_id, score in raw_scores.items() if score > 0}

    def _keyword_score(self, query: str, text: str) -> float:
        tokens = self._tokens(query)
        if not tokens:
            tokens = [query.lower()]
        haystack = text.lower()
        score = 0.0
        for token in tokens:
            count = haystack.count(token.lower())
            if count:
                score += 1.0 + math.log(count)
        return score

    def _tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
            if re.search(r"[A-Za-z0-9_]", token):
                if len(token) > 1:
                    tokens.append(token)
                continue
            if len(token) <= 4:
                tokens.append(token)
            else:
                tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
        return list(dict.fromkeys(tokens))

    def _passes_filters(
        self,
        chunk: EvidenceChunk,
        include_keywords: list[str],
        exclude_keywords: list[str],
    ) -> bool:
        text = chunk.search_text.lower()
        if include_keywords and not all(kw.lower() in text for kw in include_keywords):
            return False
        if exclude_keywords and any(kw.lower() in text for kw in exclude_keywords):
            return False
        return True

    def _citation(self, index: int, chunk: EvidenceChunk) -> Citation:
        document = self.storage.get_document(chunk.document_id)
        quote = chunk.text[:700].strip()
        return Citation(
            citation_id=f"C{index}",
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            title=document.title if document else chunk.metadata.get("title", ""),
            filename=document.filename if document else chunk.metadata.get("filename", ""),
            page=chunk.page_start + 1 if chunk.page_start is not None else None,
            section=chunk.section,
            quote=quote,
            element_ids=chunk.element_ids,
            asset_path=chunk.metadata.get("asset_path", ""),
            evidence_type=chunk.type,
        )

    def _llm_answer(
        self,
        question: str,
        chunks: list[EvidenceChunk],
        citations: list[Citation],
    ) -> str:
        api_key = Config.get_api_key()
        if not api_key:
            return ""
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=api_key,
                base_url=Config.DASHSCOPE_BASE_URL,
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
            context = "\n\n".join(
                f"[{citation.citation_id}] {citation.title} "
                f"(page {citation.page or 'unknown'}, {citation.evidence_type})\n"
                f"{chunk.text}"
                for citation, chunk in zip(citations, chunks)
            )
            prompt = (
                "You are a research assistant. Answer the question only from the cited evidence. "
                "Use citation ids like [C1]. If evidence is insufficient, say so clearly. "
                "When proposing research viewpoints, mark them as hypotheses.\n\n"
                f"Question:\n{question}\n\nEvidence:\n{context}\n\nAnswer:"
            )
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return ""

    def _extractive_answer(self, question: str, citations: list[Citation]) -> str:
        lines = ["基于当前检索证据，可以先给出以下审慎结论："]
        for citation in citations[:5]:
            quote = citation.quote.replace("\n", " ")
            lines.append(f"- [{citation.citation_id}] {quote[:220]}")
        lines.append("如果需要形成论文观点，请把以上内容视为证据线索，而不是未经核验的最终结论。")
        return "\n".join(lines)

    def _decision(self, score: float) -> str:
        if score >= 0.68:
            return "值得优先精读"
        if score >= 0.38:
            return "可作为辅助阅读"
        return "相关性较弱，先暂缓"

    def _extract_sentence(self, text: str, keywords: list[str], default: str = "") -> str:
        sentences = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
        lowered = [(sentence, sentence.lower()) for sentence in sentences if sentence.strip()]
        for sentence, low in lowered:
            if any(keyword.lower() in low for keyword in keywords):
                return sentence.strip()[:320]
        if lowered:
            return lowered[0][0].strip()[:320]
        return default or "检索证据不足，需要补充阅读。"

    def _research_ideas(self, topic: str, title: str, evidence: str) -> list[str]:
        short_topic = topic.strip()[:80]
        return [
            f"假设/研究角度：将《{title}》中的方法或结论迁移到“{short_topic}”场景，检验其边界条件。",
            f"假设/研究角度：围绕“{short_topic}”比较该文证据与其他高相关论文的一致性和冲突点。",
        ]
