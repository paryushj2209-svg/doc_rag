# Cost-Efficient Document RAG

A question-answering service for PDF, HTML, and Markdown documents. It converts documents to chunks with Docling, embeds chunks with Sentence Transformers, stores vectors and metadata in LanceDB, retrieves relevant chunks, and asks Gemini to answer using the retrieved evidence with citations. Use the CLI for ingestion and interactive questions, or run the built-in JSON HTTP service.

## Requirements

- Python 3.12 is the recommended environment for the document-processing and ML dependencies. Check the [Docling installation guide](https://docling-project.github.io/docling/getting_started/installation/) for platform-specific PyTorch setup.
- A Google Gemini API key and a model available to that key are required for answer generation.
- The first run may download Docling processing models and the configured Sentence Transformers model. Allow network access and sufficient disk space for those downloads.

## Installation

From the project root, create and activate a virtual environment, then install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency list is currently not version-pinned. For repeatable installations, pin and validate package versions for your Python and operating-system combination.

## Configuration

Set the Gemini credentials and model in the shell before running the application. The code reads environment variables directly; it does not automatically load a `.env` file.

Linux or macOS:

```bash
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="your-available-gemini-model"
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GEMINI_MODEL = "your-available-gemini-model"
```

## Usage

### Ingest documents

Ingest one or more supported files:

```bash
python src/main.py ingest ./documents/report.pdf ./documents/notes.md
```

Ingest all supported files recursively from a folder:

```bash
python src/main.py ingest ./documents
```

Supported extensions are `.pdf`, `.html`, `.htm`, `.md`, and `.markdown`. The command reports whether each file was ingested or skipped and how many chunks it produced.

The current pipeline identifies a document using its resolved file path and stores a content hash and pipeline hash in a manifest table. Re-ingesting an unchanged file with the same embedding model is skipped. Updating a file at the same path refreshes its chunk records.

Chunk size and overlap are configurable in characters (defaults: 1200 and 150). They can be set with `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`, or passed to ingestion. Changing either setting changes the pipeline fingerprint and reprocesses documents.

### Ask questions

Ask one question and exit:

```bash
python src/main.py ask "What does the report say about revenue?"
```

Start an interactive session:

```bash
python src/main.py ask
```

Choose the number of retrieved chunks:

```bash
python src/main.py ask --top-k 8 "What does the report say about revenue?"
```

The CLI converts the question to a retrieval request, embeds it with the configured Sentence Transformers model, searches LanceDB, and sends the retrieved text to Gemini. When a source name in the question uniquely matches an ingested document, the CLI applies a source filter. Answers are requested to stay grounded in retrieved evidence and cite source/chunk labels. If there are no retrieved chunks or Gemini reports that the evidence does not answer the question, the CLI reports an error.

### Run the HTTP API

```bash
python src/main.py serve --host 0.0.0.0 --port 8000
```

`GET /health` returns a status object. `POST /ask` accepts `{"question":"What does the report say about revenue?"}` and returns the cited answer, retrieved chunk count, request latency, and estimated input/output token counts. Token counts are estimates based on characters (roughly four characters per token); Gemini's provider usage metadata is not currently returned. Each request logs latency, chunk count, and estimated total tokens.

### Evaluate retrieval

Create a JSON array with 15–30 labeled cases. Each case needs a question and relevant chunk IDs (IDs are stored as `document-id:index` in LanceDB); optional `filters` and `gold_answer` fields are supported.

```json
[
  {"question":"What was the reported revenue?", "relevant_chunks":["abc123:2"], "gold_answer":"$10 million"}
]
```

Run retrieval metrics and save per-question results plus aggregate Recall@k, Hit Rate, MRR, nDCG@k, Context Precision@k, and p50/p95 retrieval latency:

```bash
python src/main.py evaluate evaluation-questions.json --top-k 5 --output evaluation-results.json
```

Add `--answers` to generate answers and score faithfulness and answer relevance with RAGAS and Gemini. `gold_answer` enables exact match and token F1. This mode makes additional Gemini evaluation calls.

### Compare monthly storage cost

```bash
python src/main.py cost --output cost-comparison.json
```

The default comparison uses 384-dimensional float32 vectors, 1 KiB metadata per vector, 30% index overhead, $0.10/GiB-month local storage, and $40/month local compute. Managed monthly amounts default to illustrative assumptions of $25/$100/$500 at 100K/1M/10M vectors. They are not vendor quotes; override them with a current quote or calculator result for the selected managed service. The calculation covers vector storage and the stated fixed local compute budget; it does not model backups, replicas, network transfer, or query-volume tiers.

| Vectors | Estimated local storage | Local total/month | Managed assumption/month |
| ---: | ---: | ---: | ---: |
| 100K | 0.31 GiB | $40.03 | $25 |
| 1M | 3.10 GiB | $40.31 | $100 |
| 10M | 30.99 GiB | $43.10 | $500 |

## Implementation overview

- `src/data/parse.py` uses Docling's `DocumentConverter` to read supported documents.
- `src/data/chunk.py` uses Docling's `HybridChunker`.
- `src/data/embedder.py` embeds chunk text with Sentence Transformers.
- `src/data/ingestion.py` stores vectors, chunk text, source information, Docling metadata, and an ingestion manifest in LanceDB.
- `src/retrieval/convert.py` preserves the user's question and recognizes explicit top-k requests and unambiguous source names.
- `src/retrieval/retrieval.py` embeds queries and performs vector search with optional source or document ID filters.
- `src/llm/llm.py` calls Gemini, requests a structured summary, and validates that citations refer to retrieved chunk IDs.
- `src/metrics.py` contains retrieval metrics, RAGAS-based answer evaluation, latency summaries, exact match/token F1 helpers, and a parameterized storage-cost estimate.


## Current limitations

- Document identity is based on the resolved path. Moving a file is treated as a new document.
- The embedding model is loaded for each file during ingestion, which adds overhead for batches of documents.
- No source corpus or relevance judgments are included, so meaningful evaluation results require a corpus-specific 15–30 question labeled dataset. The evaluator rejects smaller datasets instead of presenting a tiny sample as representative.
- Evaluation results are local artifacts and are not generated during ordinary ingestion or question answering.
- Token counts in request logs are estimates. The cost model only compares storage and stated fixed compute assumptions; it excludes query and generation costs.

## AI assistance and usage
- The implementation and the architecture was designed entirely by the person, with codebase written with some AI assistance. Documentation, metric evaluation and test runs were done by AI.
