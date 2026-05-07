import pytest
from unittest.mock import Mock, patch
import numpy as np
from rag.generator import RAGGenerator
from rag.types import DocumentChunk, RAGConfig, SearchResult
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.search import SemanticSearch


class TestRAGGenerator:
    @pytest.fixture
    def config(self):
        return RAGConfig(top_k=3, similarity_threshold=0.5)

    @pytest.fixture
    def embedder(self, config):
        return Embedder(config)

    @pytest.fixture
    def vectorstore(self, embedder):
        store = VectorStore(embedder.dimension)
        chunks = []
        for i in range(3):
            chunk = DocumentChunk(
                content=f"Test content {i}",
                metadata={"filename": f"doc{i}.txt"}
            )
            chunk.embedding = embedder.embed([chunk.content])[0]
            chunks.append(chunk)
        store.add(chunks)
        return store

    @pytest.fixture
    def search(self, embedder, vectorstore, config):
        return SemanticSearch(embedder, vectorstore, config)

    @pytest.fixture
    def generator_no_llm(self, vectorstore, search, config):
        return RAGGenerator(vectorstore, search, config, llm_client=None)

    def test_generate_without_llm_returns_context(self, generator_no_llm):
        response, sources = generator_no_llm.generate("test query")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_returns_sources(self, generator_no_llm):
        response, sources = generator_no_llm.generate("test")

        assert isinstance(sources, list)

    def test_generate_with_no_results(self, generator_no_llm):
        response, sources = generator_no_llm.generate("xyznonexistentquery")

        assert "No tengo información relevante" in response
        assert sources == []

    def test_format_response_with_sources(self, generator_no_llm):
        formatted = generator_no_llm.format_response(
            "test query",
            "Generated response",
            ["doc1.txt", "doc2.txt"]
        )

        assert "Generated response" in formatted
        assert "doc1.txt" in formatted
        assert "doc2.txt" in formatted

    def test_format_response_without_sources(self, generator_no_llm):
        formatted = generator_no_llm.format_response(
            "test query",
            "Generated response",
            []
        )

        assert "Generated response" in formatted

    def test_generate_with_llm_client(self, vectorstore, search, config):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="LLM response"))]
        mock_client.chat.completions.create.return_value = mock_response

        generator_with_llm = RAGGenerator(vectorstore, search, config, llm_client=mock_client)

        response, sources = generator_with_llm.generate("test query")

        assert "LLM response" in response
        mock_client.chat.completions.create.assert_called_once()