import pytest
import tempfile
import numpy as np
from rag.vectorstore import VectorStore
from rag.types import DocumentChunk, RAGConfig
from rag.embedder import Embedder
from rag.search import SemanticSearch


class TestVectorStore:
    @pytest.fixture
    def store(self):
        return VectorStore(dimension=384)

    @pytest.fixture
    def sample_chunks(self):
        chunks = []
        for i in range(5):
            chunk = DocumentChunk(
                content=f"Document content number {i}",
                metadata={"filename": f"doc{i}.txt", "type": "txt"}
            )
            chunk.embedding = np.random.rand(384).astype(np.float32)
            chunk.embedding = chunk.embedding / np.linalg.norm(chunk.embedding)
            chunks.append(chunk)
        return chunks

    def test_add_chunks(self, store, sample_chunks):
        store.add(sample_chunks)

        assert store.count() == 5

    def test_search_with_threshold(self, store, sample_chunks):
        store.add(sample_chunks)

        query_emb = sample_chunks[0].embedding
        results = store.search(query_emb, k=3, threshold=0.5)

        assert len(results) > 0
        assert all(r.score >= 0.5 for r in results)

    def test_search_returns_ordered_results(self, store, sample_chunks):
        store.add(sample_chunks)

        query_emb = sample_chunks[0].embedding
        results = store.search(query_emb, k=5, threshold=0.0)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load(self, store, sample_chunks, temp_dir=None):
        store.add(sample_chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded_store = VectorStore()
            loaded_store.load(tmpdir)

            assert loaded_store.count() == 5
            assert loaded_store.chunks[0].content == sample_chunks[0].content

    def test_search_empty_index(self, store):
        query_emb = np.random.rand(384).astype(np.float32)

        with pytest.raises(ValueError) as exc_info:
            store.search(query_emb, k=5, threshold=0.5)

        assert "empty" in str(exc_info.value).lower()

    def test_is_valid(self, store):
        assert not store.is_valid()

        chunks = [DocumentChunk(content="test", metadata={})]
        chunks[0].embedding = np.random.rand(384).astype(np.float32)
        store.add(chunks)

        assert store.is_valid()

    def test_threshold_filters_results(self, store, sample_chunks):
        store.add(sample_chunks)

        query_emb = sample_chunks[0].embedding
        results_high = store.search(query_emb, k=5, threshold=0.9)
        results_low = store.search(query_emb, k=5, threshold=0.1)

        assert len(results_high) <= len(results_low)