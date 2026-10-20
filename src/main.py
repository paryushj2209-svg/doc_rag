import argparse
import json
import logging
import os
import sys
import time
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


def _usage_estimate(question: str, chunks: Iterable[dict[str, Any]], answer: str) -> dict[str, int]:
    """Rough token estimate (characters / 4); provider usage metadata is not exposed."""
    contexts = "".join(str(chunk.get("text", "")) for chunk in chunks)
    input_tokens = (len(question) + len(contexts)) // 4
    output_tokens = len(answer) // 4
    return {"input": input_tokens, "output": output_tokens, "total_tokens": input_tokens + output_tokens}


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
    ingest_parser.add_argument("--chunk-size", type=int, default=int(os.getenv("RAG_CHUNK_SIZE", "1200")))
    ingest_parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("RAG_CHUNK_OVERLAP", "150")))

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
    serve_parser = subparsers.add_parser("serve", help="Run the JSON HTTP API")
    serve_parser.add_argument("--host", default=os.getenv("RAG_HOST", "127.0.0.1"))
    serve_parser.add_argument("--port", type=int, default=int(os.getenv("RAG_PORT", "8000")))
    serve_parser.add_argument("--top-k", type=int, default=5)
    serve_parser.add_argument("--db-uri", default=os.getenv("RAG_DB_URI", "./data/lancedb"))
    serve_parser.add_argument("--table", default=os.getenv("RAG_TABLE", "document_chunks"))
    serve_parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL))

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval against a labeled JSON dataset")
    eval_parser.add_argument("dataset", type=Path)
    eval_parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--answers", action="store_true", help="Also generate and judge answers (uses Gemini/RAGAS)")
    eval_parser.add_argument("--db-uri", default=os.getenv("RAG_DB_URI", "./data/lancedb"))
    eval_parser.add_argument("--table", default=os.getenv("RAG_TABLE", "document_chunks"))
    eval_parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL))

    cost_parser = subparsers.add_parser("cost", help="Build an assumption-based monthly storage cost comparison")
    cost_parser.add_argument("--output", type=Path, default=Path("cost-comparison.json"))
    cost_parser.add_argument("--dimensions", type=int, default=384)
    cost_parser.add_argument("--local-storage-usd-per-gib-month", type=float, default=0.10)
    cost_parser.add_argument("--local-compute-usd-per-month", type=float, default=40.0)
    cost_parser.add_argument("--managed-100k", type=float, default=25.0)
    cost_parser.add_argument("--managed-1m", type=float, default=100.0)
    cost_parser.add_argument("--managed-10m", type=float, default=500.0)
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
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            _print_ingestion_results(results)
            return 0

        if args.command == "evaluate":
            from metrics import run_evaluation

            dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
            if not isinstance(dataset, list) or not 15 <= len(dataset) <= 30:
                raise ValueError("The evaluation dataset must be a JSON array with 15–30 questions")
            active_retriever = retriever or LanceDBRetriever(
                database_uri=args.db_uri,
                table_name=args.table,
                embedding_model=args.embedding_model,
            )
            answer_generator = None
            if args.answers:
                answer_generator = lambda question, chunks: summarize_retrieved(
                    {"query": question}, chunks, model=os.getenv("GEMINI_MODEL")
                )
            results = run_evaluation(dataset, active_retriever, k=args.top_k, answer_generator=answer_generator)
            args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Wrote {len(dataset)} evaluation results to {args.output}")
            print(json.dumps(results["aggregate"], indent=2))
            return 0

        if args.command == "cost":
            from metrics import monthly_cost_comparison

            result = monthly_cost_comparison(
                {100_000: args.managed_100k, 1_000_000: args.managed_1m, 10_000_000: args.managed_10m},
                dimensions=args.dimensions,
                local_storage_usd_per_gib_month=args.local_storage_usd_per_gib_month,
                local_compute_usd_per_month=args.local_compute_usd_per_month,
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote cost comparison to {args.output}")
            for row in result["rows"]:
                print(f"{row['vectors']:,} vectors: local ${row['local_monthly_usd']:.2f}/mo; managed assumption ${row['managed_monthly_usd']:.2f}/mo")
            return 0

        if args.command == "serve":
            if not 1 <= args.top_k <= 100 or not 1 <= args.port <= 65535:
                raise ValueError("top_k must be 1–100 and port must be 1–65535")
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            active_retriever = retriever or LanceDBRetriever(
                database_uri=args.db_uri,
                table_name=args.table,
                embedding_model=args.embedding_model,
            )
            available_sources = source_loader(args.db_uri) if source_loader else []

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path != "/health":
                        self.send_error(404)
                        return
                    self._json(200, {"status": "ok"})

                def do_POST(self):
                    if self.path != "/ask":
                        self.send_error(404)
                        return
                    started = time.perf_counter()
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length < 1 or length > 1_000_000:
                            raise ValueError("Request body must be between 1 byte and 1 MB")
                        payload = json.loads(self.rfile.read(length))
                        question = payload.get("question") if isinstance(payload, dict) else None
                        if not isinstance(question, str) or not question.strip():
                            raise ValueError("JSON body must contain a non-empty 'question'")
                        request = convert_query(question, available_sources, args.top_k)
                        chunks = active_retriever.retrieve(request)
                        answer_text = summarize_retrieved(request, chunks, model=os.getenv("GEMINI_MODEL"))
                        elapsed = (time.perf_counter() - started) * 1000
                        usage = _usage_estimate(question, chunks, answer_text)
                        logging.info("request_ms=%.2f chunks=%d estimated_tokens=%d", elapsed, len(chunks), usage["total_tokens"])
                        self._json(200, {"answer": answer_text, "chunk_count": len(chunks), "latency_ms": round(elapsed, 2), "estimated_tokens": usage})
                    except (ValueError, ImportError, RuntimeError, json.JSONDecodeError) as exc:
                        self._json(400, {"error": str(exc)})

                def _json(self, status, payload):
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, fmt, *values):
                    logging.info("%s - %s", self.address_string(), fmt % values)

            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            server = ThreadingHTTPServer((args.host, args.port), Handler)
            print(f"Serving RAG API at http://{args.host}:{args.port} (POST /ask, GET /health)")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
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
