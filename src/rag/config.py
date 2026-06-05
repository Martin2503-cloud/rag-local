"""
Configuración del sistema RAG.

Provee funciones para cargar la configuración default y crear instancias
de RAGConfig con los valores apropiados para el entorno.
"""
import os
from pathlib import Path
from rag.types import RAGConfig


def get_default_config() -> RAGConfig:
    """
    Retorna la configuración por defecto del sistema RAG.
    
    Los valores default están optimizados para documentación técnica:
    - chunk_size=750: Balance entre contexto y precisión
    - chunk_overlap=112 (15%): Preserva continuidad semántica entre chunks
    - all-MiniLM-L6-v2: Modelo pequeño (80MB) y rápido, sin API key
    - similarity_threshold=0.35: Balance entre recall y precisión
    - top_k=4: Suficientes resultados diversos
    
    Returns:
        RAGConfig con valores por defecto.
    """
    return RAGConfig(
        chunk_size=750,
        chunk_overlap=112,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.35,
        top_k=4,
        index_path=str(Path("data/index").resolve()),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


def load_config() -> RAGConfig:
    """
    Carga la configuración del sistema.
    
    Currently retorna la configuración default. En el futuro puede
    extenderse para leer desde archivo YAML o variables de entorno.
    
    Returns:
        RAGConfig instanciada con los valores de configuración.
    """
    return get_default_config()