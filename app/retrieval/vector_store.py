
import numpy as np


class VectorStore:
    def __init__(self):
        self.chunks: list[dict] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

    def add(self, chunks: list[dict], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if len(chunks) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        if self.vectors.shape[0] > 0:
            if embeddings.shape[1] != self.vectors.shape[1]:
                raise ValueError("Embedding dimensions do not match.")
            self.vectors = np.vstack([self.vectors, embeddings])
        else:
            self.vectors = embeddings.copy()

        self.chunks.extend(dict(chunk) for chunk in chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not self.chunks:
            return []

        query_embedding = np.asarray(
            query_embedding, dtype=np.float32
        ).reshape(-1)

        if query_embedding.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query embedding dimensions do not match.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values.")

        if np.linalg.norm(query_embedding) == 0:
            raise ValueError("Query embedding must not be a zero vector.")

        scores = self.vectors @ query_embedding
        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in best_indices:
            result = dict(self.chunks[int(index)])
            result["score"] = float(scores[index])
            results.append(result)

        return results
