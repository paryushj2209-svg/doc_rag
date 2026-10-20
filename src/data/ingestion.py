import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from data.chunk import chunk_document
from data.embedder import DEFAULT_MODEL, embed_chunks
from data.parse import parse_file


@dataclass(frozen=True)
class IngestionResult:
    source: str
    status: str
    chunk_count: int


class IngestionPipeline:
    def __init__(
        self,
        database_uri: str = "./data/lancedb",
        table_name: str = "document_chunks",
        manifest_table_name: str = "document_manifest",
        embedding_model: str = DEFAULT_MODEL,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        db: Any = None,
        parser: Callable[[str | Path], Any] = parse_file,
        chunker: Callable[..., list[Any]] = chunk_document,
        embedder: Callable[..., list[list[float]]] = embed_chunks,
    ) -> None:
        if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and chunk_overlap must be smaller")
        self.database_uri = database_uri
        self.table_name = table_name
        self.manifest_table_name = manifest_table_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._db = db
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder

    def _database(self) -> Any:
        if self._db is None:
            try:
                import lancedb
            except ImportError as exc:
                raise ImportError(
                    "LanceDB is required for ingestion. Install dependencies from requirements.txt."
                ) from exc
            self._db = lancedb.connect(self.database_uri)
        return self._db

    def _table(self, name: str, initial_rows: list[dict[str, Any]] | None = None) -> Any:
        db = self._database()
        if name in db.table_names():
            return db.open_table(name)
        if not initial_rows:
            return None
        return db.create_table(name, data=initial_rows)

    def _pipeline_hash(self) -> str:
        settings = {

            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _document_id(source: Path) -> str:
        return hashlib.sha256(str(source).encode("utf-8")).hexdigest()

    @staticmethod
    def _file_hash(source: Path) -> str:
        return hashlib.sha256(source.read_bytes()).hexdigest()

    @staticmethod
    def _metadata(chunk: Any) -> dict[str, Any]:
        metadata = getattr(getattr(chunk, "meta", None), "export_json_dict", None)
        if callable(metadata):
            exported = metadata()
            if isinstance(exported, dict):
                return exported
        return {}

    def _existing_manifest(self, document_id: str) -> dict[str, Any] | None:
        table = self._table(self.manifest_table_name)
        if table is None:
            return None
        rows = table.search().where(f"document_id = '{document_id}'").limit(1).to_list()
        return rows[0] if rows else None

    @staticmethod
    def _upsert(table: Any, key: str, rows: list[dict[str, Any]]) -> None:
        (table.merge_insert(key)
         .when_matched_update_all()
         .when_not_matched_insert_all()
         .execute(rows))

    def _store_chunks(self, document_id: str, records: list[dict[str, Any]]) -> None:
        table = self._table(self.table_name, records)
        if table is None:
            return
        if not records:
            table.delete(f"document_id = '{document_id}'")
            return
        (table.merge_insert("chunk_id")
         .when_matched_update_all()
         .when_not_matched_insert_all()
         .when_not_matched_by_source_delete(f"document_id = '{document_id}'")
         .execute(records))

    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        source = Path(file_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Source is not a file: {source}")

        document_id = self._document_id(source)
        file_hash = self._file_hash(source)
        pipeline_hash = self._pipeline_hash()
        manifest = self._existing_manifest(document_id)
        if manifest and manifest.get("file_hash") == file_hash and manifest.get("pipeline_hash") == pipeline_hash:
            return IngestionResult(str(source), "skipped", int(manifest.get("chunk_count", 0)))

        document = self._parser(source)
        chunks = self._chunker(document, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        vectors = self._embedder(chunks, model_name=self.embedding_model)
        if len(chunks) != len(vectors):
            raise ValueError("The embedder must return one vector per chunk")

        records: list[dict[str, Any]] = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_metadata = self._metadata(chunk)
            records.append(
                {
                    "chunk_id": f"{document_id}:{index}",
                    "document_id": document_id,
                    "source": str(source),
                    "chunk_index": index,
                    "text": chunk.text,
                    "vector": vector,
                    "metadata_json": json.dumps(chunk_metadata, sort_keys=True, default=str),
                    "file_hash": file_hash,
                    "pipeline_hash": pipeline_hash,
                }
            )

        self._store_chunks(document_id, records)
        manifest_record = {
            "document_id": document_id,
            "source": str(source),
            "file_hash": file_hash,
            "pipeline_hash": pipeline_hash,
            "chunk_count": len(records),
        }
        manifest_table = self._table(self.manifest_table_name, [manifest_record])
        if manifest_table is not None and manifest_table.count_rows() > 0:
            self._upsert(manifest_table, "document_id", [manifest_record])
        return IngestionResult(str(source), "ingested", len(records))

    def ingest_files(self, file_paths: Iterable[str | Path]) -> list[IngestionResult]:
        return [self.ingest_file(file_path) for file_path in file_paths]


def ingest_files(
    file_paths: Iterable[str | Path],
    database_uri: str = "./data/lancedb",
    embedding_model: str = DEFAULT_MODEL,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[IngestionResult]:
    pipeline = IngestionPipeline(
        database_uri=database_uri,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return pipeline.ingest_files(file_paths)
