from pathlib import Path

from docling.document_converter import DocumentConverter


def parse_file(file_path: str | Path):
    result = DocumentConverter().convert(source=str(file_path))
    return result.document
