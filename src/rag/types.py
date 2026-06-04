"""
Definiciones de tipos fundamentales para el sistema RAG.

Este módulo contiene las clases de datos que representan los conceptos
核心 del sistema de búsqueda semántica: documentos, chunks, y configuración.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DocumentChunk:
    """
    Representa un fragmento de documento con su contenido y metadatos.
    
    Attributes:
        content: Texto extraído del documento original.
        metadata: Diccionario con metadatos del chunk (filename, tipo, página, etc.).
        embedding: Vector numpy con el embedding denso del conteúdo (384 dimensiones).
    """
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """
    Resultado de una búsqueda semántica con el chunk encontrado y su puntuación.
    
    Attributes:
        chunk: El DocumentChunk que matcheó con la query.
        score: Puntuación de similitud del coseno (0-1, donde 1 es idéntico).
        source: Nombre del archivo fuente del chunk.
    """
    chunk: DocumentChunk
    score: float
    source: str


@dataclass
class RAGConfig:
    """
    Configuración centralizada del sistema RAG.
    
    Attributes:
        chunk_size: Tamaño máximo de cada chunk en caracteres (default: 750).
        chunk_overlap: Superposición entre chunks consecutivos (default: 112 = 15%).
        embedding_model: Modelo de embeddings a usar (default: sentence-transformers/all-MiniLM-L6-v2).
        similarity_threshold: Umbral mínimo de similitud para retornar resultados (default: 0.7).
        top_k: Número máximo de resultados a retornar por búsqueda (default: 4).
        index_path: Ruta donde se persiste el índice vectorial (default: data/index).
        openai_api_key: Clave API de OpenAI para generación de respuestas (opcional).
    """
    chunk_size: int = 750
    chunk_overlap: int = 112
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.35
    top_k: int = 4
    index_path: str = "data/index"
    openai_api_key: Optional[str] = None