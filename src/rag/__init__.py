"""
RAG Local - Buscador Semántico con Retrieval-Augmented Generation.

Sistema de búsqueda semántica local que permite consultar documentación
técnica usando lenguaje natural. No requiere API externa (usa embeddings
locales) y puede usar OpenAI para generación de respuestas.

Módulos:
- types: Definiciones de tipos (DocumentChunk, SearchResult, RAGConfig)
- config: Configuración del sistema
- loader: Cargadores de documentos (PDF, MD, TXT)
- chunker: División de texto en chunks
- embedder: Generación de embeddings
- vectorstore: Persistencia vectorial con FAISS
- search: Búsqueda semántica
- generator: Generación de respuestas RAG
- cli: Interfaz CLI y clase RAGSystem

Quick start:
    >>> from rag import RAGSystem
    >>> rag = RAGSystem()
    >>> rag.ingest("documento.pdf")
    >>> rag.load_index()
    >>> response = rag.query("qué contiene el documento")
"""
from rag.types import DocumentChunk, SearchResult, RAGConfig
from rag import config
from rag.loader import DocumentLoader
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.search import SemanticSearch
from rag.generator import RAGGenerator
from rag.cli import RAGSystem

__all__ = [
    "DocumentChunk",
    "SearchResult",
    "RAGConfig",
    "config",
    "DocumentLoader",
    "TextChunker",
    "Embedder",
    "VectorStore",
    "SemanticSearch",
    "RAGGenerator",
    "RAGSystem",
]