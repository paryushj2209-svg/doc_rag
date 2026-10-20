import json
import re
from pathlib import PurePosixPath
from typing import Iterable


_TOP_K_PATTERNS = (
    re.compile(r"\btop\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+(?:results?|chunks?)\b", re.IGNORECASE),
)


def _validate_top_k(value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"top_k must be an integer between 1 and {maximum}")


def _requested_top_k(prompt: str, default: int, maximum: int) -> int:
    values = {
        int(match.group(1))
        for pattern in _TOP_K_PATTERNS
        for match in pattern.finditer(prompt)
    }
    if not values:
        return default
    if len(values) > 1:
        return default
    value = values.pop()
    _validate_top_k(value, maximum)
    return value


def _source_filter(prompt: str, available_sources: Iterable[str]) -> str | None:
    normalized_prompt = prompt.casefold()
    matches: set[tuple[int, str]] = set()

    for source in available_sources:
        value = str(source).strip()
        if not value:
            continue
        normalized_source = value.replace("\\", "/")
        aliases = {value, PurePosixPath(normalized_source).name}
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            expression = rf"(?<![\w]){re.escape(alias.casefold())}(?![\w])"
            if re.search(expression, normalized_prompt):
                matches.add((len(alias), value))

    if not matches:
        return None
    longest_alias = max(length for length, _ in matches)
    sources = {source for length, source in matches if length == longest_alias}
    return next(iter(sources)) if len(sources) == 1 else None


def convert_query(
    prompt: str,
    available_sources: Iterable[str] = (),
    default_top_k: int = 5,
    maximum_top_k: int = 100,
) -> dict[str, object]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a non-empty question")
    _validate_top_k(maximum_top_k, 100)
    _validate_top_k(default_top_k, maximum_top_k)

    question = prompt.strip()
    filters: dict[str, str] = {}
    source = _source_filter(question, available_sources)
    if source is not None:
        filters["source"] = source

    return {
        "query": question,
        "top_k": _requested_top_k(question, default_top_k, maximum_top_k),
        "filters": filters,
    }


def query_to_json(
    prompt: str,
    available_sources: Iterable[str] = (),
    default_top_k: int = 5,
    maximum_top_k: int = 100,
) -> str:
    request = convert_query(
        prompt,
        available_sources=available_sources,
        default_top_k=default_top_k,
        maximum_top_k=maximum_top_k,
    )
    return json.dumps(request, ensure_ascii=False, indent=2)
