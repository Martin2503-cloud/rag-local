"""
Generación de embeddings vectoriales para búsqueda semántica.

Este módulo maneja la conversión de texto a vectores densos (embeddings) usando
sentence-transformers. Los embeddings capturan el significado semántico
del texto en un espacio vectorial de alta dimensionalidad.

Modelo default: all-MiniLM-L6-v2
- Dimensionalidad: 384
- Tamaño: ~80MB
- Rápido y eficiente, no requiere API key
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.types import DocumentChunk, RAGConfig


class Embedder:
    """
    Genera embeddings vectoriales para textos usando sentence-transformers.
    
    Convierte texto en vectores densos de 384 dimensiones que capturan
    el significado semántico. El modelo default es all-MiniLM-L6-v2,
    optimizado para velocidad sin sacrificar demasiada calidad.
    
    Attributes:
        config: Configuración con el modelo de embeddings a usar.
        model: Instancia del modelo de sentence-transformers.
        dimension: Dimensionalidad del espaço de embeddings (384).
    
    Ejemplo de uso:
        >>> embedder = Embedder(config)
        >>> query_emb = embedder.embed_query("cómo funciona el sistema")
        >>> print(query_emb.shape)  # (384,)
    """
    
    def __init__(self, config: RAGConfig):
        """
        Inicializa el embedder con el modelo configurado.
        
        Args:
            config: RAGConfig con embedding_model especificado.
        """
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        try:
            self.dimension = self.model.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.
        
        Procesa los textos en batch para eficiencia. Los embeddings
        son normalizados (L2) para permitir búsqueda por cosIP.
        
        Args:
            texts: Lista de strings a convertir en embeddings.
            
        Returns:
            Array numpy de forma (n, 384) con embeddings normalizados.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Genera embedding para una query del usuario.
        
        Args:
            query: String con la consulta del usuario.
            
        Returns:
            Array numpy de forma (384,) con embedding normalizado.
        """
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0]

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Genera embeddings para una lista de chunks de documento.
        
        Actualiza cada chunk con su embedding correspondiente.
        
        Args:
            chunks: Lista de DocumentChunk a embeber.
            
        Returns:
            Lista de chunks con el campo embedding populated.
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks