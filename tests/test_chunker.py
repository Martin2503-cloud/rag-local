import pytest
from rag.chunker import TextChunker
from rag.types import DocumentChunk, RAGConfig


class TestTextChunker:
    @pytest.fixture
    def config(self):
        return RAGConfig(chunk_size=50, chunk_overlap=10)

    @pytest.fixture
    def chunker(self, config):
        return TextChunker(config)

    def test_chunk_documents(self, chunker):
        docs = [
            DocumentChunk(content="This is a long text that needs to be divided into smaller chunks for processing."),
            DocumentChunk(content="Another document here.")
        ]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) > len(docs)
        for chunk in chunks:
            assert len(chunk.content) <= 50

    def test_chunk_preserves_metadata(self, chunker):
        docs = [
            DocumentChunk(
                content="Short text.",
                metadata={"filename": "test.md", "type": "md"}
            )
        ]

        chunks = chunker.chunk_documents(docs)

        assert chunks[0].metadata["filename"] == "test.md"
        assert chunks[0].metadata["type"] == "md"

    def test_chunk_single_document(self, chunker):
        docs = [
            DocumentChunk(content="Short text.")
        ]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) >= 1
        assert chunks[0].content == "Short text."