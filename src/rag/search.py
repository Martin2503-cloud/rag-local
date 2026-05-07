"""
Motor de búsqueda semántica.

Este módulo proporciona la capa de abstracción para realizar búsquedas
usando el pipeline completo: query → embedding → FAISS search.

La búsqueda semántica difiere de la búsqueda por palabras clave
en que encuentra documentos conceptualmente relacionados aunque
no contengan las palabras exactas de la query.
"""
from typing import List
import numpy as np
from rag.types import DocumentChunk, SearchResult, RAGConfig
from rag.embedder import Embedder
from rag.vectorstore import VectorStore


class SemanticSearch:
    """
    Motor de búsqueda semántica que orquesta el pipeline completo.
    
    Coordina el embedder y el vectorstore para realizar búsquedas:
    1. Convierte la query del usuario en embedding
    2. Busca en el índice vectorial
    3. Retorna resultados con scores de similitud
    
    Attributes:
        embedder: Instancia para generar embeddings.
        vectorstore: Índice vectorial con los documentos.
        config: Configuración con top_k y threshold.
    
    Example:
        >>> search = SemanticSearch(embedder, vectorstore, config)
        >>> results = search.search("cómo funciona la autenticación")
        >>> for r in results:
        >>>     print(f"{r.source}: {r.score:.3f}")
    """
    
    def __init__(self, embedder: Embedder, vectorstore: VectorStore, config: RAGConfig):
        """
        Inicializa el motor de búsqueda.
        
        Args:
            embedder: Instancia configurada para generar embeddings.
            vectorstore: Índice con documentos ingestados.
            config: Configuración de búsqueda.
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.config = config

    def search(self, query: str) -> List[SearchResult]:
        """
        Busca documentos similiares a la query.
        
        Convierte la query a embedding usando el mismo modelo
        que se usó para indexar los documentos, luego
        busca en FAISS los k documentos más cercanos.
        
        Args:
            query: String con la consulta del usuario.
            
        Returns:
            Lista de SearchResult ordenados por score descendente.
            
        Raises:
            ValueError: Si no hay documentos indexados.
        """
        if not self.vectorstore.is_valid():
            raise ValueError("Index is empty. Please ingest documents first.")
        
        query_emb = self.embedder.embed_query(query)
        
        return self.vectorstore.search(
            query_emb=query_emb,
            k=self.config.top_k,
            threshold=self.config.similarity_threshold,
        )

    def search_with_context(self, query: str) -> tuple[str, List[SearchResult]]:
        """
        Busca y formatea el contexto para el LLM.
        
        Combina los chunks encontrados en un string de contexto
        legible para inyección en el prompt del LLM.
        Formato: [filename] contenido del chunk
        
        Args:
            query: String con la consulta del usuario.
            
        Returns:
            Tupla (context_string, resultados).
        """
        results = self.search(query)
        
        if not results:
            context = "No se encontró información relevante."
        else:
            context = "\n\n".join([
                f"[{r.source}] {r.chunk.content}"
                for r in results
            ])
        
        return context, results