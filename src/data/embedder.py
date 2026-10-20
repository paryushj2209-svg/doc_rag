from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_chunks(chunks, model_name: str = DEFAULT_MODEL):
    model = SentenceTransformer(model_name)
    texts = [chunk.text for chunk in chunks]
    return model.encode(texts, convert_to_numpy=True).tolist()
