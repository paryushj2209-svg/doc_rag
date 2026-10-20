import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from data.embedder import DEFAULT_MODEL
from data.ingestion import IngestionResult, ingest_files
from llm.llm import summarize_retrieved
from retrieval.convert import convert_query
from retrieval.retrieval import LanceDBRetriever


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}


def collect_documents(paths: Iterable[str | Path]) -> list[Path]:
    documents: set[Path] = set()
    for item in paths:
        path = Path(item).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported document type: {path}")
            documents.add(path.resolve())
        elif path.is_dir():
            documents.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            raise ValueError(f"Path is not a regular file or directory: {path}")
    return sorted(documents)


def load_available_sources(database_uri: str) -> list[str]:
    try:
        import lancedb
    except ImportError as exc:
        raise ImportError("LanceDB is required. Install dependencies from requirements.txt.") from exc

    database = lancedb.connect(database_uri)
    if "document_manifest" not in database.table_names():
        return []
    rows = database.open_table("document_manifest").search().select(["source"]).to_list()
    return sorted({row["source"] for row in rows if row.get("source")})


def format_evidence(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant document chunks were found."

    passages = []
    for result in results:
        source = result.get("source", "unknown source")
        chunk_index = result.get("chunk_index", "?")
        passages.append(f"[{source} | chunk {chunk_index}]\n{result.get('text', '')}")
    return "Retrieved evidence:\n\n" + "\n\n".join(passages)


def ask_documents(
    question: str,
    retriever: Any,
    top_k: int = 5,
    available_sources: Iterable[str] = (),
) -> str:
    request = convert_query(
        question,
        available_sources=available_sources,
        default_top_k=top_k,
    )
    results = retriever.retrieve(request)
    return summarize_retrieved(request, results, model=os.getenv("GEMINI_MODEL"))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("paths", nargs="+", type=Path)
    ingest_parser.add_argument(
        "--db-uri",
        default=os.getenv("RAG_DB_URI", "./data/lancedb"),
    )
    ingest_parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL),
    )

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question", nargs="*")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument(
        "--db-uri",
        default=os.getenv("RAG_DB_URI", "./data/lancedb"),
    )
    ask_parser.add_argument(
        "--table",
        default=os.getenv("RAG_TABLE", "document_chunks"),
    )
    ask_parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL),
    )
    return parser


def _print_ingestion_results(results: Iterable[IngestionResult]) -> None:
    for result in results:
        print(f"{result.status}: {result.source} ({result.chunk_count} chunks)")


def main(
    argv: list[str] | None = None,
    retriever: Any = None,
    ingestion_runner: Callable[..., list[IngestionResult]] = ingest_files,
    source_loader: Callable[[str], list[str]] = load_available_sources,
) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "ingest":
            documents = collect_documents(args.paths)
            if not documents:
                raise ValueError("No supported PDF, HTML, or Markdown documents were found")
            results = ingestion_runner(
                documents,
                database_uri=args.db_uri,
                embedding_model=args.embedding_model,
            )
            _print_ingestion_results(results)
            return 0

        if not 1 <= args.top_k <= 100:
            raise ValueError("--top-k must be between 1 and 100")
        active_retriever = retriever or LanceDBRetriever(
            database_uri=args.db_uri,
            table_name=args.table,
            embedding_model=args.embedding_model,
        )
        available_sources = source_loader(args.db_uri) if source_loader else []

        def answer(question: str) -> bool:
            try:
                print(ask_documents(question, active_retriever, args.top_k, available_sources))
                return True
            except (ImportError, ValueError, RuntimeError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return False

        if args.question:
            return 0 if answer(" ".join(args.question)) else 2

        print("Ask a question about your documents. Enter 'exit' to quit.")
        while True:
            try:
                question = input("Question: ").strip()
            except EOFError:
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                answer(question)
        return 0
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
