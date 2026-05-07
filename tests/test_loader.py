import pytest
import tempfile
import os
from pathlib import Path
from rag.loader import DocumentLoader


class TestDocumentLoader:
    @pytest.fixture
    def loader(self):
        return DocumentLoader()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_markdown(self, loader, temp_dir):
        md_path = os.path.join(temp_dir, "test.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Test Document\n\nThis is content.")

        chunks = loader.load(md_path)

        assert len(chunks) == 1
        assert "Test Document" in chunks[0].content
        assert chunks[0].metadata["type"] == "md"
        assert chunks[0].metadata["filename"] == "test.md"

    def test_load_text(self, loader, temp_dir):
        txt_path = os.path.join(temp_dir, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Plain text content here.")

        chunks = loader.load(txt_path)

        assert len(chunks) == 1
        assert chunks[0].content == "Plain text content here."
        assert chunks[0].metadata["type"] == "txt"

    def test_load_pdf(self, loader, temp_dir):
        import fitz
        pdf_path = os.path.join(temp_dir, "test.pdf")
        doc = fitz.open()
        page = doc.new_page(width=100, height=100)
        page.insert_text((10, 10), "PDF Content")
        doc.save(pdf_path)
        doc.close()

        chunks = loader.load(pdf_path)

        assert len(chunks) == 1
        assert "PDF Content" in chunks[0].content
        assert chunks[0].metadata["type"] == "pdf"
        assert chunks[0].metadata["page"] == 1

    def test_unsupported_format(self, loader, temp_dir):
        ext_path = os.path.join(temp_dir, "test.xyz")

        with pytest.raises(ValueError) as exc_info:
            loader.load(ext_path)

        assert "Unsupported format" in str(exc_info.value)

    def test_supported_formats(self, loader):
        formats = loader.supported_formats()

        assert ".pdf" in formats
        assert ".md" in formats
        assert ".txt" in formats