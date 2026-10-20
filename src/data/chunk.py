from docling.chunking import HybridChunker


def chunk_document(document, chunk_size: int = 1200, chunk_overlap: int = 150):
    """Create semantically grouped Docling chunks, then apply bounded text windows.

    Docling's chunker handles layout and tables; this final pass makes the exposed
    size and overlap settings predictable for ordinary text chunks as well.
    """
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller")
    result = []
    step = chunk_size - chunk_overlap
    for original in HybridChunker().chunk(dl_doc=document):
        text = original.text
        if len(text) <= chunk_size:
            result.append(original)
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Prefer a nearby word boundary without allowing a tiny window.
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            chunk_text = text[start:end].strip()
            if hasattr(original, "model_copy"):
                chunk = original.model_copy(update={"text": chunk_text})
            else:
                import copy

                chunk = copy.copy(original)
                chunk.text = chunk_text
            result.append(chunk)
            if end == len(text):
                break
            start = max(start + 1, end - chunk_overlap)
    return result
