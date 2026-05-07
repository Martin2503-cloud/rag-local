import pytest
import numpy as np
from rag.embedder import Embedder
from rag.types import DocumentChunk, RAGConfig


class TestEmbedder:
    @pytest.fixture
    def config(self):
        return RAGConfig(embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    @pytest.fixture
    def embedder(self, config):
        return Embedder(config)

    def test_embed_dimension(self, embedder):
        text = "This is a test sentence."

        embedding = embedder.embed([text])

        assert embedding.shape[1] == 384

    def test_embed_single_text(self, embedder):
        text = "Single text embedding"

        embeddings = embedder.embed([text])

        assert embeddings.shape == (1, 384)
        assert np.allclose(np.linalg.norm(embeddings), 1.0, atol=1e-5)

    def test_embed_batch(self, embedder):
        texts = ["Text one", "Text two", "Text three"]

        embeddings = embedder.embed(texts)

        assert embeddings.shape == (3, 384)

    def test_embed_query(self, embedder):
        query = "What is the meaning of life?"

        embedding = embedder.embed_query(query)

        assert embedding.shape == (384,)
        assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_embed_chunks(self, embedder):
        chunks = [
            DocumentChunk(content="First chunk"),
            DocumentChunk(content="Second chunk"),
        ]

        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert result[0].embedding is not None
        assert result[1].embedding is not None
        assert result[0].embedding.shape == (384,)

    def test_deterministic_embedding(self, embedder):
        text = "Same text should give same embedding"

        emb1 = embedder.embed([text])
        emb2 = embedder.embed([text])

        assert np.allclose(emb1, emb2)