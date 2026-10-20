import asyncio
import math
import os
import re
import string
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _relevance_map(relevant_chunks: Sequence[str] | Mapping[str, float]) -> dict[str, float]:
    if isinstance(relevant_chunks, Mapping):
        relevance = {str(chunk_id): float(score) for chunk_id, score in relevant_chunks.items()}
    elif isinstance(relevant_chunks, Sequence) and not isinstance(relevant_chunks, (str, bytes)):
        relevance = {str(chunk_id): 1.0 for chunk_id in relevant_chunks}
    else:
        raise TypeError("relevant_chunks must be a sequence of chunk IDs or a relevance mapping")
    if any(not math.isfinite(score) or score < 0 for score in relevance.values()):
        raise ValueError("Relevance scores must be finite, non-negative numbers")
    return relevance


def retrieval_metrics(
    retrieved_chunks: Sequence[Mapping[str, Any]],
    relevant_chunks: Sequence[str] | Mapping[str, float],
    k: int = 5,
) -> dict[str, float | int]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    relevance = _relevance_map(relevant_chunks)
    ranked = list(retrieved_chunks[:k])
    grades = [relevance.get(str(chunk.get("chunk_id", "")), 0.0) for chunk in ranked]
    relevant_count = sum(score > 0 for score in relevance.values())
    found_count = sum(score > 0 for score in grades)
    reciprocal_rank = next((1.0 / rank for rank, score in enumerate(grades, 1) if score > 0), 0.0)

    average_precision = 0.0
    relevant_seen = 0
    for rank, score in enumerate(grades, 1):
        if score > 0:
            relevant_seen += 1
            average_precision += relevant_seen / rank
    if relevant_count:
        average_precision /= min(relevant_count, k)

    def dcg(scores: Sequence[float]) -> float:
        return sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(scores, 1))

    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_grades)
    ndcg = dcg(grades) / ideal_dcg if ideal_dcg else 0.0

    return {
        f"recall@{k}": found_count / relevant_count if relevant_count else 0.0,
        "hit_rate": float(found_count > 0),
        "mrr": reciprocal_rank,
        f"ndcg@{k}": ndcg,
        f"context_precision@{k}": average_precision,
    }


def normalize_answer(answer: str) -> str:
    normalized = unicodedata.normalize("NFKD", answer).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = "".join(character for character in normalized if character not in string.punctuation)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def exact_match_and_f1(prediction: str, gold_answer: str) -> dict[str, float]:
    predicted = normalize_answer(prediction)
    gold = normalize_answer(gold_answer)
    if predicted == gold:
        return {"exact_match": 1.0, "token_f1": 1.0}
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    if not predicted_tokens or not gold_tokens:
        return {"exact_match": 0.0, "token_f1": 0.0}
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return {"exact_match": 0.0, "token_f1": 0.0}
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"exact_match": 0.0, "token_f1": f1}


def create_ragas_answer_judge(
    model: str | None = None,
    embedding_model: str | None = None,
    metrics: tuple[Any, Any] | None = None,
) -> Callable[[str, str, Sequence[Mapping[str, Any]]], dict[str, float]]:
    if metrics is None:
        selected_model = model or os.getenv("GEMINI_MODEL")
        api_key = os.getenv("GEMINI_API_KEY")
        if not selected_model:
            raise ValueError("Set GEMINI_MODEL to use the RAGAS answer evaluator")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY to use the RAGAS answer evaluator")
        try:
            from google import genai
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.llms import llm_factory
            from ragas.metrics.collections import AnswerRelevancy, Faithfulness
            from src.data.embedder import DEFAULT_MODEL
        except ImportError as exc:
            raise ImportError("Install the 'ragas' dependency and its Google GenAI integration to evaluate answers") from exc
        client = genai.Client(api_key=api_key)
        evaluator_llm = llm_factory(selected_model, provider="google", client=client)
        evaluator_embeddings = HuggingFaceEmbeddings(model=embedding_model or DEFAULT_MODEL)
        metrics = (
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        )

    faithfulness_metric, relevancy_metric = metrics

    def judge(
        question: str,
        answer: str,
        retrieved_chunks: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:
        contexts = [str(chunk.get("text", "")) for chunk in retrieved_chunks]

        async def score_metrics() -> tuple[Any, Any]:
            faithfulness, relevance = await asyncio.gather(
                faithfulness_metric.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
                relevancy_metric.ascore(user_input=question, response=answer),
            )
            return faithfulness, relevance

        faithfulness, relevance = asyncio.run(score_metrics())
        return {
            "faithfulness": float(faithfulness.value),
            "answer_relevance": float(relevance.value),
        }

    return judge


def evaluate_answer(
    question: str,
    answer: str,
    retrieved_chunks: Sequence[Mapping[str, Any]],
    gold_answer: str | None = None,
    judge: Callable[..., Mapping[str, float]] | None = None,
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {
        "faithfulness": None,
        "answer_relevance": None,
        "exact_match": None,
        "token_f1": None,
    }
    if judge is not None:
        judged = judge(question, answer, retrieved_chunks)
        for name in ("faithfulness", "answer_relevance"):
            score = judged.get(name)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValueError(f"Judge must return a {name} score between 0 and 1")
            scores[name] = float(score)
    if gold_answer is not None:
        scores.update(exact_match_and_f1(answer, gold_answer))
    return scores


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("At least one latency measurement is required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_evaluation(
    dataset: Sequence[Mapping[str, Any]],
    retriever: Any,
    k: int = 5,
    answer_generator: Callable[[str, Sequence[Mapping[str, Any]]], str] | None = None,
    judge: Callable[..., Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    if not dataset:
        raise ValueError("The evaluation dataset must contain at least one question")
    if answer_generator is not None and judge is None:
        judge = create_ragas_answer_judge()
    per_question = []
    latencies = []
    for case in dataset:
        question = case.get("question", case.get("query"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Each evaluation case must contain a non-empty question")
        relevant = case.get("relevant_chunks", case.get("relevant_chunk_ids"))
        if relevant is None:
            raise ValueError("Each evaluation case must contain relevant_chunks")
        request = {"query": question, "top_k": k, "filters": case.get("filters", {})}
        started = time.perf_counter()
        retrieved = retriever.retrieve(request)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        row: dict[str, Any] = {
            "question": question,
            "retrieval": retrieval_metrics(retrieved, relevant, k),
            "retrieval_latency_ms": latency_ms,
        }
        if answer_generator is not None:
            answer = answer_generator(question, retrieved)
            if not isinstance(answer, str):
                raise TypeError("answer_generator must return a string")
            row["answer"] = evaluate_answer(
                question,
                answer,
                retrieved,
                gold_answer=case.get("gold_answer"),
                judge=judge,
            )
        per_question.append(row)

    retrieval_keys = per_question[0]["retrieval"].keys()
    aggregate: dict[str, float] = {
        metric: sum(row["retrieval"][metric] for row in per_question) / len(per_question)
        for metric in retrieval_keys
    }
    answer_rows = [row["answer"] for row in per_question if "answer" in row]
    if answer_rows:
        for metric in answer_rows[0]:
            available = [row[metric] for row in answer_rows if row[metric] is not None]
            if available:
                aggregate[metric] = sum(available) / len(available)

    return {
        "question_count": len(per_question),
        "k": k,
        "aggregate": aggregate,
        "retrieval_latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "per_question": per_question,
    }


def monthly_cost_comparison(
    managed_monthly_usd: Mapping[int, float | None],
    *,
    dimensions: int,
    local_storage_usd_per_gib_month: float,
    local_compute_usd_per_month: float,
    metadata_bytes_per_vector: int = 1024,
    index_overhead_ratio: float = 0.3,
    vector_counts: Sequence[int] = (100_000, 1_000_000, 10_000_000),
) -> dict[str, Any]:
    if dimensions < 1 or metadata_bytes_per_vector < 0 or index_overhead_ratio < 0:
        raise ValueError("dimensions must be positive and storage overheads non-negative")
    if local_storage_usd_per_gib_month < 0 or local_compute_usd_per_month < 0:
        raise ValueError("Local monthly costs cannot be negative")
    bytes_per_vector = dimensions * 4 + metadata_bytes_per_vector
    rows = []
    for count in vector_counts:
        if count < 1:
            raise ValueError("vector_counts must contain positive integers")
        storage_gib = count * bytes_per_vector * (1 + index_overhead_ratio) / (1024**3)
        rows.append({
            "vectors": count,
            "estimated_storage_gib": storage_gib,
            "local_monthly_usd": local_compute_usd_per_month + storage_gib * local_storage_usd_per_gib_month,
            "managed_monthly_usd": managed_monthly_usd.get(count),
        })
    return {
        "managed_service": "Qdrant Cloud",
        "assumptions": {
            "dimensions": dimensions,
            "bytes_per_component": 4,
            "metadata_bytes_per_vector": metadata_bytes_per_vector,
            "index_overhead_ratio": index_overhead_ratio,
            "local_storage_usd_per_gib_month": local_storage_usd_per_gib_month,
            "local_compute_usd_per_month": local_compute_usd_per_month,
            "managed_monthly_usd": "Supply Qdrant Cloud quote/calculator result for each scale; null means unknown.",
        },
        "rows": rows,
    }
