
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str):
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
