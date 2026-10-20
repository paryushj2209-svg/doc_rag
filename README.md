# Document RAG CLI

A command-line question-answering application for PDF, HTML, and Markdown documents. It converts documents to chunks with Docling, embeds chunks with Sentence Transformers, stores vectors and metadata in LanceDB, retrieves relevant chunks, and asks Gemini to answer using the retrieved evidence with citations.

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

Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DB_URI` | `./data/lancedb` | LanceDB database directory |
| `RAG_TABLE` | `document_chunks` | Vector table used by retrieval |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers model used for document and query embeddings |

The embedding model used at ingestion must match the model used at retrieval. Keep the same `RAG_EMBEDDING_MODEL` value for both operations. LanceDB data is local and excluded from Git.

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

### Inspect Gemini's raw response

The diagnostic script uses the same query conversion and LanceDB retrieval path but returns Gemini's response text without requiring or parsing JSON:

```bash
python src/test/test_gemini_raw.py
```

It prompts for a question. It requires an already-ingested corpus, the configuration above, and a live Gemini API call.

## Implementation overview

- `src/data/parse.py` uses Docling's `DocumentConverter` to read supported documents.
- `src/data/chunk.py` uses Docling's `HybridChunker`.
- `src/data/embedder.py` embeds chunk text with Sentence Transformers.
- `src/data/ingestion.py` stores vectors, chunk text, source information, Docling metadata, and an ingestion manifest in LanceDB.
- `src/retrieval/convert.py` preserves the user's question and recognizes explicit top-k requests and unambiguous source names.
- `src/retrieval/retrieval.py` embeds queries and performs vector search with optional source or document ID filters.
- `src/llm/llm.py` calls Gemini, requests a structured summary, and validates that citations refer to retrieved chunk IDs.
- `src/metrics.py` contains retrieval metrics, RAGAS-based answer evaluation, latency summaries, exact match/token F1 helpers, and a parameterized storage-cost estimate.
- `src/test/test.py` contains mocked unit tests. `src/test/test_gemini_raw.py` is a manual live-API diagnostic script.

LlamaIndex packages are present in `requirements.txt`, but the current `src/data/ingestion.py` implementation uses Docling and the project's own ingestion code directly; it does not currently call LlamaIndex.

## Tests

The test module is named `src/test/test.py`, so specify it explicitly when running pytest:

```bash
python -m pytest src/test/test.py -q
```

Most tests use mocks and do not call Gemini. The raw-response diagnostic script does make a real Gemini API call and is separate from the unit tests.

## Evaluation

The metrics are currently exposed as Python functions in `src/metrics.py`; there is no evaluation CLI, fixed question dataset, or generated results file yet. Retrieval evaluation requires each test question to have relevant chunk IDs or graded relevance values. Answer evaluation can use RAGAS Faithfulness and AnswerRelevancy with Gemini and local Hugging Face embeddings; this is an evaluation-only path and does not add calls to ordinary RAG requests. The cost helper requires explicit local-cost assumptions and managed-service prices and does not fetch live pricing.

## Current limitations

- Chunk size and overlap are not currently configurable; `HybridChunker` uses its defaults.
- The pipeline hash currently includes the embedding model but not parser/chunker settings or dependency versions. Include all ingestion-affecting settings in the hash before changing chunking behavior and relying on re-ingestion skips.
- Document identity is based on the resolved path. Moving a file is treated as a new document.
- The embedding model is loaded for each file during ingestion, which adds overhead for batches of documents.
- Dependencies are not pinned, and the current implementation does not have an HTTP service or a ready-to-run evaluation dataset/results artifact.

