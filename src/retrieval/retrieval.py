import json
from typing import Any, Mapping

from data.embedder import DEFAULT_MODEL


FILTER_COLUMNS = {"document_id", "source"}


class LanceDBRetriever:
    def __init__(
        self,
        database_uri: str = "./data/lancedb",
        table_name: str = "document_chunks",
        embedding_model: str = DEFAULT_MODEL,
        db: Any = None,
        model: Any = None,
    ) -> None:
        self.database_uri = database_uri
        self.table_name = table_name
        self.embedding_model = embedding_model
        self._db = db
        self._model = model

    def _database(self) -> Any:
        if self._db is None:
            try:
                import lancedb
            except ImportError as exc:
                raise ImportError(
                    "LanceDB is required for retrieval. Install dependencies from requirements.txt."
                ) from exc
            self._db = lancedb.connect(self.database_uri)
        return self._db

    def _embedding_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    @staticmethod
    def _normalize_request(request: str | Mapping[str, Any]) -> tuple[str, int, dict[str, str]]:
        if isinstance(request, str):
            data: Mapping[str, Any] = {"query": request}
        elif isinstance(request, Mapping):
            data = request
        else:
            raise TypeError("The retrieval request must be a string or mapping")

        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The retrieval request must include a non-empty 'query'")

        top_k = data.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")

        filters = data.get("filters", {})
        if not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping")
        if not set(filters).issubset(FILTER_COLUMNS):
            raise ValueError(f"Supported filters are: {', '.join(sorted(FILTER_COLUMNS))}")
        if any(not isinstance(value, str) or not value for value in filters.values()):
            raise ValueError("Filter values must be non-empty strings")
        return query.strip(), top_k, dict(filters)

    def retrieve(self, request: str | Mapping[str, Any]) -> list[dict[str, Any]]:
        query, top_k, filters = self._normalize_request(request)
        vector = self._embedding_model().encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].tolist()

        table = self._database().open_table(self.table_name)
        search = table.search(vector).limit(top_k)
        if filters:
            expressions = [
                f"{column} = '{value.replace(chr(39), chr(39) + chr(39))}'"
                for column, value in filters.items()
            ]
            search = search.where(" AND ".join(expressions), prefilter=True)

        matches = search.to_list()
        results: list[dict[str, Any]] = []
        for match in matches:
            record = dict(match)
            metadata_json = record.pop("metadata_json", "{}")
            try:
                record["metadata"] = json.loads(metadata_json) if metadata_json else {}
            except (TypeError, json.JSONDecodeError):
                record["metadata"] = {}
            results.append(record)
        return results
