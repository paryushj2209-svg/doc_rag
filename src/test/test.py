import json
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.chunk import chunk_document
from src.data.embedder import DEFAULT_MODEL, embed_chunks
from src.data.ingestion import IngestionPipeline
from src.data.parse import parse_file
from src.llm.llm import NoSupportingInformationError, summarize_retrieved
from src.main import ask_documents, collect_documents, main
from src.metrics import create_ragas_answer_judge, evaluate_answer, monthly_cost_comparison, retrieval_metrics, run_evaluation
from src.retrieval.convert import convert_query, query_to_json
from src.retrieval.retrieval import LanceDBRetriever
from src.test.test_gemini_raw import ask as ask_raw_gemini


class MetricsTests(unittest.TestCase):
    def test_ragas_answer_judge_scores_faithfulness_and_relevance(self):
        faithfulness = MagicMock()
        relevancy = MagicMock()

        async def faithfulness_score(**kwargs):
            self.assertEqual(kwargs["retrieved_contexts"], ["Evidence text."])
            return SimpleNamespace(value=0.9)

        async def relevancy_score(**kwargs):
            self.assertEqual(kwargs["user_input"], "Question?")
            return SimpleNamespace(value=0.8)

        faithfulness.ascore.side_effect = faithfulness_score
        relevancy.ascore.side_effect = relevancy_score
        judge = create_ragas_answer_judge(metrics=(faithfulness, relevancy))
        self.assertEqual(
            judge("Question?", "Answer.", [{"text": "Evidence text."}]),
            {"faithfulness": 0.9, "answer_relevance": 0.8},
        )
        faithfulness.ascore.assert_called_once()
        relevancy.ascore.assert_called_once()

    def test_retrieval_metrics_use_relevance_and_rank(self):
        scores = retrieval_metrics(
            [{"chunk_id": "noise"}, {"chunk_id": "a"}, {"chunk_id": "b"}],
            {"a": 3, "b": 1},
            k=3,
        )
        self.assertEqual(scores["recall@3"], 1.0)
        self.assertEqual(scores["hit_rate"], 1.0)
        self.assertEqual(scores["mrr"], 0.5)
        self.assertLess(scores["ndcg@3"], 1.0)
        self.assertAlmostEqual(scores["context_precision@3"], 0.5833333333)

    def test_answer_metrics_include_judge_and_optional_gold_scores(self):
        scores = evaluate_answer(
            "What happened?",
            "Revenue increased",
            [{"text": "Revenue increased."}],
            gold_answer="The revenue increased",
            judge=lambda question, answer, chunks: {
                "faithfulness": 0.9,
                "answer_relevance": 0.8,
            },
        )
        self.assertEqual(scores["faithfulness"], 0.9)
        self.assertEqual(scores["answer_relevance"], 0.8)
        self.assertEqual(scores["exact_match"], 1.0)
        self.assertEqual(scores["token_f1"], 1.0)

    def test_evaluation_reports_latency_and_aggregate_retrieval_scores(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [{"chunk_id": "c1", "text": "Evidence."}]
        results = run_evaluation(
            [{"question": "Question?", "relevant_chunks": ["c1"]}],
            retriever,
            k=1,
        )
        self.assertEqual(results["question_count"], 1)
        self.assertEqual(results["aggregate"]["recall@1"], 1.0)
        self.assertGreaterEqual(results["retrieval_latency_ms"]["p95"], 0)

    def test_cost_comparison_applies_declared_storage_assumptions(self):
        comparison = monthly_cost_comparison(
            {100_000: 0.0, 1_000_000: None, 10_000_000: 125.0},
            dimensions=384,
            local_storage_usd_per_gib_month=1.0,
            local_compute_usd_per_month=0.0,
            metadata_bytes_per_vector=0,
            index_overhead_ratio=0.0,
        )
        self.assertAlmostEqual(comparison["rows"][0]["estimated_storage_gib"], 0.143051, places=5)
        self.assertEqual(comparison["rows"][0]["managed_monthly_usd"], 0.0)
        self.assertEqual(comparison["rows"][2]["managed_monthly_usd"], 125.0)


class ParseTests(unittest.TestCase):
    @patch("src.data.parse.DocumentConverter")
    def test_parse_file_returns_docling_document(self, converter_class):
        document = object()
        converter_class.return_value.convert.return_value.document = document
        self.assertIs(parse_file("sample.pdf"), document)
        converter_class.return_value.convert.assert_called_once_with(source="sample.pdf")


class ChunkTests(unittest.TestCase):
    @patch("src.data.chunk.HybridChunker")
    def test_chunk_document_uses_docling_hybrid_chunker(self, chunker_class):
        document = object()
        expected_chunks = [object(), object()]
        chunker_class.return_value.chunk.return_value = iter(expected_chunks)
        self.assertEqual(chunk_document(document), expected_chunks)
        chunker_class.assert_called_once_with()
        chunker_class.return_value.chunk.assert_called_once_with(dl_doc=document)


class EmbedTests(unittest.TestCase):
    @patch("src.data.embedder.SentenceTransformer")
    def test_embed_chunks_encodes_chunk_texts(self, model_class):
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        model_class.return_value = model
        chunks = [MagicMock(text="chunk text")]
        self.assertEqual(embed_chunks(chunks), [[0.1, 0.2]])
        model_class.assert_called_once_with(DEFAULT_MODEL)
        model.encode.assert_called_once_with(["chunk text"], convert_to_numpy=True)


class FakeMergeInsert:
    def __init__(self, table, key):
        self.table = table
        self.key = key
        self.delete_condition = None

    def when_matched_update_all(self):
        return self

    def when_not_matched_insert_all(self):
        return self

    def when_not_matched_by_source_delete(self, condition):
        self.delete_condition = condition
        return self

    def execute(self, rows):
        incoming_ids = {row[self.key] for row in rows}
        if self.delete_condition:
            document_id = self.delete_condition.split("'", 2)[1]
            self.table.rows = [
                row for row in self.table.rows
                if row.get("document_id") != document_id or row.get(self.key) in incoming_ids
            ]
        for row in rows:
            self.table.rows = [
                existing for existing in self.table.rows
                if existing.get(self.key) != row[self.key]
            ]
            self.table.rows.append(row)


class FakeTable:
    def __init__(self, rows):
        self.rows = list(rows)
        self.condition = None

    def search(self):
        return self

    def where(self, condition):
        self.condition = condition
        return self

    def limit(self, count):
        return self

    def to_list(self):
        document_id = self.condition.split("'", 2)[1]
        return [row for row in self.rows if row.get("document_id") == document_id]

    def merge_insert(self, key):
        return FakeMergeInsert(self, key)

    def count_rows(self):
        return len(self.rows)

    def delete(self, condition):
        document_id = condition.split("'", 2)[1]
        self.rows = [row for row in self.rows if row.get("document_id") != document_id]


class FakeDatabase:
    def __init__(self):
        self.tables = {}

    def table_names(self):
        return list(self.tables)

    def create_table(self, name, data):
        self.tables[name] = FakeTable(data)
        return self.tables[name]

    def open_table(self, name):
        return self.tables[name]


class IngestionTests(unittest.TestCase):
    def test_reingestion_is_idempotent_and_stores_metadata(self):
        node = SimpleNamespace(
            text="chunk text",
            metadata={"page": 3},
            embedding=[0.1, 0.2],
        )
        database = FakeDatabase()
        node_processor = MagicMock(return_value=[node])
        pipeline = IngestionPipeline(
            db=database,
            node_processor=node_processor,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.md"
            path.write_text("first version", encoding="utf-8")
            first = pipeline.ingest_file(path)
            second = pipeline.ingest_file(path)

        chunk_rows = database.tables["document_chunks"].rows
        self.assertEqual(first.status, "ingested")
        self.assertEqual(second.status, "skipped")
        self.assertEqual(len(chunk_rows), 1)
        self.assertEqual(chunk_rows[0]["text"], "chunk text")
        self.assertEqual(chunk_rows[0]["vector"], [0.1, 0.2])
        self.assertEqual(chunk_rows[0]["metadata_json"], '{"page": 3}')
        node_processor.assert_called_once()


class QueryConversionTests(unittest.TestCase):
    def test_preserves_question_and_matches_known_source(self):
        request = convert_query(
            "  What does the budget say in Plan.pdf?  ",
            available_sources=["Plan.pdf", "report.md"],
        )
        self.assertEqual(request, {
            "query": "What does the budget say in Plan.pdf?",
            "top_k": 5,
            "filters": {"source": "Plan.pdf"},
        })

    def test_uses_explicit_result_count(self):
        request = convert_query("Show the top 8 chunks about staffing")
        self.assertEqual(request["top_k"], 8)

    def test_ambiguous_source_match_adds_no_filter(self):
        request = convert_query(
            "Summarize plan.pdf",
            available_sources=["team-a/plan.pdf", "team-b/plan.pdf"],
        )
        self.assertEqual(request["filters"], {})

    def test_unmatched_source_adds_no_filter(self):
        request = convert_query(
            "What is the policy?",
            available_sources=["plan.pdf"],
        )
        self.assertEqual(request["filters"], {})

    def test_conflicting_result_counts_use_default(self):
        request = convert_query("Show top 3 results and top 8 chunks")
        self.assertEqual(request["top_k"], 5)

    def test_rejects_empty_prompt_and_out_of_range_result_count(self):
        with self.assertRaises(ValueError):
            convert_query("  ")
        with self.assertRaises(ValueError):
            convert_query("Show top 101 results")

    def test_returns_readable_json(self):
        encoded = query_to_json("What is in plan.pdf?", ["plan.pdf"])
        self.assertEqual(json.loads(encoded)["filters"], {"source": "plan.pdf"})


class MainCliTests(unittest.TestCase):
    def test_raw_gemini_ask_returns_api_text_without_json_parsing(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [{
            "chunk_id": "guide:0",
            "source": "guide.md",
            "chunk_index": 0,
            "text": "The guide describes the process.",
        }]
        client = MagicMock()
        raw_text = "Here is the answer in plain text, without JSON."
        client.models.generate_content.return_value = SimpleNamespace(text=raw_text)

        response = ask_raw_gemini(
            "What process is described in guide.md?",
            retriever=retriever,
            available_sources=["guide.md"],
            client=client,
            model="raw-test-model",
        )

        self.assertEqual(response, raw_text)
        retriever.retrieve.assert_called_once_with({
            "query": "What process is described in guide.md?",
            "top_k": 5,
            "filters": {"source": "guide.md"},
        })
        api_call = client.models.generate_content.call_args.kwargs
        self.assertEqual(api_call["model"], "raw-test-model")
        self.assertIn("The guide describes the process.", api_call["contents"])
        self.assertNotIn("response_json_schema", api_call["config"])
        self.assertNotIn("response_mime_type", api_call["config"])

    def test_collect_documents_recursively_selects_supported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.pdf").touch()
            (root / "notes.md").touch()
            (root / "ignore.txt").touch()
            (root / "nested").mkdir()
            (root / "nested" / "page.html").touch()
            collected = collect_documents([root])
        self.assertEqual(
            {path.name for path in collected},
            {"one.pdf", "notes.md", "page.html"},
        )

    @patch.dict("os.environ", {"GEMINI_MODEL": "configured-model"})
    @patch("src.main.summarize_retrieved", return_value="Summarized answer.\n\nSources: /docs/plan.pdf | chunk 1")
    def test_ask_documents_passes_structured_query_to_summarizer(self, summarize):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"chunk_id": "plan:1", "source": "/docs/plan.pdf", "chunk_index": 1, "text": "Budget details."}
        ]
        output = ask_documents(
            "What is the budget in plan.pdf?",
            retriever,
            top_k=4,
            available_sources=["/docs/plan.pdf"],
        )
        self.assertEqual(output, "Summarized answer.\n\nSources: /docs/plan.pdf | chunk 1")
        retriever.retrieve.assert_called_once_with({
            "query": "What is the budget in plan.pdf?",
            "top_k": 4,
            "filters": {"source": "/docs/plan.pdf"},
        })
        summarize.assert_called_once_with(
            retriever.retrieve.call_args.args[0],
            retriever.retrieve.return_value,
            model="configured-model",
        )

    def test_ingest_command_passes_folder_files_to_ingestion(self):
        ingestion_runner = MagicMock(return_value=[])
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "doc.md").touch()
            status = main(
                ["ingest", directory],
                ingestion_runner=ingestion_runner,
            )
        self.assertEqual(status, 0)
        args, kwargs = ingestion_runner.call_args
        self.assertEqual([path.name for path in args[0]], ["doc.md"])
        self.assertEqual(kwargs["database_uri"], "./data/lancedb")

    @patch("src.main.summarize_retrieved", return_value="The answer is here.\n\nSources: guide.md | chunk 0")
    def test_ask_command_runs_one_shot_question(self, summarize):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"chunk_id": "guide:0", "source": "guide.md", "chunk_index": 0, "text": "The answer is here."}
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                ["ask", "What", "is", "the", "answer?"],
                retriever=retriever,
                source_loader=lambda database_uri: ["guide.md"],
            )
        self.assertEqual(status, 0)
        self.assertIn("The answer is here.", output.getvalue())
        self.assertEqual(retriever.retrieve.call_args.args[0]["query"], "What is the answer?")
        summarize.assert_called_once()

    @patch("src.main.summarize_retrieved", side_effect=[NoSupportingInformationError("No relevant information"), "Supported answer."])
    @patch("builtins.input", side_effect=["unsupported question", "supported question", "exit"])
    def test_interactive_ask_continues_after_question_error(self, input_mock, summarize):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"chunk_id": "guide:0", "source": "guide.md", "chunk_index": 0, "text": "Evidence."}
        ]
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), patch("sys.stderr", errors):
            status = main(
                ["ask"],
                retriever=retriever,
                source_loader=lambda database_uri: ["guide.md"],
            )
        self.assertEqual(status, 0)
        self.assertIn("Supported answer.", output.getvalue())
        self.assertIn("No relevant information", errors.getvalue())
        self.assertEqual(summarize.call_count, 2)


class FakeVector:
    def tolist(self):
        return [0.1, 0.2]


class FakeEncodedVectors:
    def __getitem__(self, index):
        return FakeVector()


class FakeSearch:
    def __init__(self, results):
        self.results = results
        self.filter_expression = None
        self.limit_value = None
        self.prefilter = None

    def limit(self, value):
        self.limit_value = value
        return self

    def where(self, expression, prefilter=False):
        self.filter_expression = expression
        self.prefilter = prefilter
        return self

    def to_list(self):
        return self.results


class FakeVectorTable:
    def __init__(self, results):
        self.results = results
        self.query_vector = None
        self.search_builder = None

    def search(self, vector):
        self.query_vector = vector
        self.search_builder = FakeSearch(self.results)
        return self.search_builder


class RetrievalTests(unittest.TestCase):
    def test_retrieval_embeds_query_filters_and_returns_metadata(self):
        results = [{
            "source": "report.pdf",
            "chunk_index": 4,
            "text": "Evidence text",
            "metadata_json": '{"page": 5}',
            "_distance": 0.2,
        }]
        table = FakeVectorTable(results)
        db = MagicMock()
        db.open_table.return_value = table
        model = MagicMock()
        model.encode.return_value = FakeEncodedVectors()
        retriever = LanceDBRetriever(db=db, model=model)

        matches = retriever.retrieve({
            "query": "Find the evidence",
            "top_k": 3,
            "filters": {"source": "report.pdf"},
        })

        self.assertEqual(table.query_vector, [0.1, 0.2])
        self.assertEqual(table.search_builder.limit_value, 3)
        self.assertEqual(table.search_builder.filter_expression, "source = 'report.pdf'")
        self.assertTrue(table.search_builder.prefilter)
        self.assertEqual(matches[0]["metadata"], {"page": 5})
        self.assertNotIn("metadata_json", matches[0])

    def test_retrieval_rejects_unknown_filter_fields(self):
        retriever = LanceDBRetriever(db=MagicMock(), model=MagicMock())
        with self.assertRaises(ValueError):
            retriever.retrieve({"query": "question", "filters": {"page": "5"}})


class LlmTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps({
                "status": "ok",
                "summary": "The report says revenue increased.",
                "citations": ["doc-1:2"],
            })
        )
        self.request = {"query": "What happened to revenue?", "top_k": 2, "filters": {}}
        self.chunks = [{
            "chunk_id": "doc-1:2",
            "source": "report.pdf",
            "chunk_index": 2,
            "text": "Revenue increased during the period.",
        }]

    def test_summarizes_only_valid_retrieved_chunks_with_citation(self):
        summary = summarize_retrieved(
            self.request,
            self.chunks,
            client=self.client,
            model="gemini-3.8-flash-lite",
        )
        self.assertEqual(
            summary,
            "The report says revenue increased.\n\nSources: report.pdf | chunk 2",
        )
        call = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(call["model"], "gemini-3.8-flash-lite")
        self.assertEqual(call["config"]["response_json_schema"]["properties"]["status"]["enum"], ["ok", "error"])
        self.assertEqual(call["config"]["temperature"], 0)

    @patch.dict("os.environ", {"GEMINI_MODEL": "gemini-3.8-flash-lite"})
    def test_uses_model_from_environment_when_model_is_not_passed(self):
        summarize_retrieved(self.request, self.chunks, client=self.client)
        self.assertEqual(self.client.models.generate_content.call_args.kwargs["model"], "gemini-3.8-flash-lite")

    @patch.dict("os.environ", {"GEMINI_MODEL": "environment-model"})
    def test_explicit_model_takes_precedence_over_environment(self):
        summarize_retrieved(self.request, self.chunks, client=self.client, model="explicit-model")
        self.assertEqual(self.client.models.generate_content.call_args.kwargs["model"], "explicit-model")

    @patch.dict("os.environ", {}, clear=True)
    def test_requires_model_when_argument_and_environment_are_missing(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_MODEL"):
            summarize_retrieved(self.request, self.chunks, client=self.client)
        self.client.models.generate_content.assert_not_called()

    def test_empty_chunks_and_null_chunk_id_fail_before_model_call(self):
        with self.assertRaises(NoSupportingInformationError):
            summarize_retrieved(self.request, [], client=self.client, model="model")
        invalid_chunk = dict(self.chunks[0], chunk_id=None)
        with self.assertRaisesRegex(ValueError, "chunk_id"):
            summarize_retrieved(self.request, [invalid_chunk], client=self.client, model="model")
        self.client.models.generate_content.assert_not_called()

    def test_model_reports_error_when_evidence_does_not_answer_question(self):
        self.client.models.generate_content.return_value.text = json.dumps({
            "status": "error",
            "summary": "The retrieved documents do not contain this information.",
            "citations": [],
        })
        with self.assertRaisesRegex(NoSupportingInformationError, "do not contain"):
            summarize_retrieved(
                self.request,
                self.chunks,
                client=self.client,
                model="model",
            )

    def test_rejects_citations_not_present_in_retrieved_chunks(self):
        self.client.models.generate_content.return_value.text = json.dumps({
            "status": "ok",
            "summary": "A summary.",
            "citations": ["unretrieved:0"],
        })
        with self.assertRaisesRegex(RuntimeError, "not retrieved"):
            summarize_retrieved(
                self.request,
                self.chunks,
                client=self.client,
                model="model",
            )


if __name__ == "__main__":
    unittest.main()
