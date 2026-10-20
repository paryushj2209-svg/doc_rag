from docling.chunking import HybridChunker


def chunk_document(document):
    return list(HybridChunker().chunk(dl_doc=document))
