import pytest
import numpy as np
from rag.types import DocumentChunk, RAGConfig, SearchResult
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.search import SemanticSearch


class TestSemanticSearch:
    @pytest.fixture
    def config(self):
        return RAGConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=3,
            similarity_threshold=0.5
        )

    @pytest.fixture
    def embedder(self, config):
        return Embedder(config)

    @pytest.fixture
    def vectorstore(self, embedder):
        store = VectorStore(embedder.dimension)
        chunks = []
        for i in range(5):
            chunk = DocumentChunk(
                content=f"Content about topic {i} with some searchable text",
                metadata={"filename": f"doc{i}.txt", "type": "txt"}
            )
            chunk.embedding = embedder.embed([chunk.content])[0]
            chunks.append(chunk)
        store.add(chunks)
        return store

    @pytest.fixture
    def search(self, embedder, vectorstore, config):
        return SemanticSearch(embedder, vectorstore, config)

    def test_search_returns_results(self, search):
        results = search.search("topic 0 content")

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_with_context(self, search):
        context, results = search.search_with_context("topic 1")

        assert isinstance(context, str)
        assert len(results) > 0

    def test_search_empty_index(self):
        config = RAGConfig()
        embedder = Embedder(config)
        empty_store = VectorStore(embedder.dimension)
        search = SemanticSearch(embedder, empty_store, config)

        with pytest.raises(ValueError) as exc_info:
            search.search("query")

        assert "empty" in str(exc_info.value).lower()

    def test_search_respects_top_k(self, search):
        results = search.search("topic")

        assert len(results) <= 3

    def test_search_threshold_filters(self, search):
        results = search.search("topic")

        assert all(r.score >= 0.5 for r in results)