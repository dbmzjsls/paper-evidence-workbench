from __future__ import annotations

import argparse
import json
import sys

from src.config import Config
from src.indexing import IndexingService, get_db_stats
from src.retrieval import ResearchRAG


def main():
    parser = argparse.ArgumentParser(description="Research paper retrieval workbench")
    parser.add_argument("--question", "-q", help="Backward-compatible one-shot query")
    parser.add_argument("--stats", action="store_true", help="Show corpus stats")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild corpus from data dir")
    parser.add_argument(
        "--mode",
        default="ensemble",
        choices=["vector", "ensemble", "rerank", "multiquery"],
        help="Compatibility option; retrieval is citation-aware hybrid by default",
    )

    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Parse and index papers")
    ingest.add_argument("--path", default=Config.DATA_DIR, help="File or directory to ingest")
    ingest.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirs")
    ingest.add_argument("--force", action="store_true", help="Reparse duplicate files")

    query = subparsers.add_parser("query", help="Ask a citation-aware question")
    query.add_argument("question")
    query.add_argument("--mode", default="ensemble")

    screen = subparsers.add_parser("screen", help="Screen papers for a research topic")
    screen.add_argument("topic")
    screen.add_argument("--limit", type=int, default=Config.SCREEN_DEFAULT_LIMIT)
    screen.add_argument("--include", nargs="*", default=[])
    screen.add_argument("--exclude", nargs="*", default=[])

    subparsers.add_parser("stats", help="Show corpus stats")

    args = parser.parse_args()
    Config.ensure_dirs()

    if args.command == "ingest":
        job = IndexingService().ingest_path(
            args.path,
            recursive=not args.no_recursive,
            force=args.force,
        )
        print(json.dumps(_dump(job), ensure_ascii=False, indent=2))
        return

    if args.command == "query":
        result = _answer_or_exit(args.question, mode=args.mode)
        print(json.dumps(_dump(result), ensure_ascii=False, indent=2))
        return

    if args.command == "screen":
        report = ResearchRAG().screen(
            args.topic,
            limit=args.limit,
            include_keywords=args.include,
            exclude_keywords=args.exclude,
        )
        print(json.dumps(_dump(report), ensure_ascii=False, indent=2))
        return

    if args.command == "stats" or args.stats:
        print(json.dumps(get_db_stats(), ensure_ascii=False, indent=2))
        if not args.question and not args.rebuild:
            return

    if args.rebuild:
        job = IndexingService().rebuild_corpus(Config.DATA_DIR)
        print(json.dumps(_dump(job), ensure_ascii=False, indent=2))
        if not args.question:
            return

    if args.question:
        result = _answer_or_exit(args.question, mode=args.mode)
        print(result.answer)
        if result.citations:
            print("\nCitations:")
            for citation in result.citations:
                page = f", page {citation.page}" if citation.page else ""
                print(f"- [{citation.citation_id}] {citation.title}{page}: {citation.quote[:160]}")
        return

    parser.print_help()


def _answer_or_exit(question: str, mode: str):
    try:
        return ResearchRAG().answer(question, mode=mode)
    except Exception as exc:
        print(f"查询失败: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _dump(model_or_dict):
    if hasattr(model_or_dict, "model_dump"):
        return model_or_dict.model_dump()
    if hasattr(model_or_dict, "dict"):
        return model_or_dict.dict()
    return model_or_dict


if __name__ == "__main__":
    main()
