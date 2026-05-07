"""
Almacenamiento vectorial usando FAISS con persistencia a disco.

Este módulo proporciona el VectorStore que gestiona el índice vectorial de FAISS
y su persistencia. FAISS (Facebook AI Similarity Search) es una biblioteca
eficiente para búsqueda de similaridad en espacios vectoriales de alta dimensionalidad.

La persistencia incluye:
- vectors.index: Archivo binario con el índice FAISS
- metadata.json: Contenido y metadatos de cada chunk

Tipo de índice: IndexFlatIP (Inner Product con normalización = Cosine Similarity)
"""
import json
import pickle
from pathlib import Path
from typing import List
import numpy as np
import faiss
from rag.types import DocumentChunk, SearchResult


class VectorStore:
    """
    Gestor del índice vectorial con persistencia.
    
    Maneja la creación, búsqueda y persistencia del índice de embeddings
    usando FAISS. Usa búsqueda por Inner Product normalizado que es
    equivalente a similaridad del coseno.
    
    Attributes:
        dimension: Dimensionalidad de los vectores (384 para all-MiniLM-L6-v2).
        index: Índice FAISS en memoria.
        chunks: Lista de chunks correspondientes a cada vector en el índice.
    
    Example:
        >>> store = VectorStore(384)
        >>> store.add(chunks_with_embeddings)
        >>> store.save("data/index")
        >>> # En otra sesión:
        >>> store.load("data/index")
        >>> results = store.search(query_emb, k=4, threshold=0.7)
    """
    
    def __init__(self, dimension: int = 384):
        """
        Inicializa el VectorStore.
        
        Args:
            dimension: Dimensionalidad de los vectores de embedding.
        """
        self.dimension = dimension
        self.index: faiss.Index = None
        self.chunks: List[DocumentChunk] = []

    def add(self, chunks: List[DocumentChunk]) -> None:
        """
        Agrega chunks al índice vectorial.
        
        Los chunks deben tener el campo embedding ya populado.
        Los vectores se normalizan para permitir búsqueda por cosIP.
        
        Args:
            chunks: Lista de DocumentChunk con embeddings calculados.
        """
        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def save(self, path: str) -> None:
        """
        Persiste el índice a disco.
        
        Crea dos archivos en el directorio especificado:
        - vectors.index: Índice binario de FAISS
        - metadata.json: Contenido y metadatos de cada chunk
        
        Args:
            path: Directorio donde guardar el índice.
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, str(path_obj / "vectors.index"))
        
        metadata = [
            {"content": chunk.content, "metadata": chunk.metadata}
            for chunk in self.chunks
        ]
        with open(path_obj / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        Carga un índice persistido desde disco.
        
        Args:
            path: Directorio donde está almacenado el índice.
            
        Raises:
            FileNotFoundError: Si no existen los archivos necesarios.
        """
        path_obj = Path(path)
        index_file = path_obj / "vectors.index"
        metadata_file = path_obj / "metadata.json"
        
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        
        self.index = faiss.read_index(str(index_file))
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.chunks = [
            DocumentChunk(content=m["content"], metadata=m["metadata"])
            for m in metadata
        ]
        
        self.dimension = self.index.d

    def search(self, query_emb: np.ndarray, k: int, threshold: float) -> List[SearchResult]:
        """
        Busca los k documentos más similares a la query.
        
        Usa búsqueda por Inner Product normalizado (equivalente a coseno).
        Aplica un threshold para filtrar resultados no relevantes.
        
        Args:
            query_emb: Embedding de la query (shape: (384,)).
            k: Número máximo de resultados a retornar.
            threshold: Threshold mínimo de similitud (0-1).
            
        Returns:
            Lista de SearchResult ordenados por score descendente.
            
        Raises:
            ValueError: Si el índice está vacío.
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty or not loaded")
        
        query = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if dist >= threshold:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(dist),
                    source=self.chunks[idx].metadata.get("filename", "unknown"),
                ))
        
        return results

    def is_valid(self) -> bool:
        """
        Verifica si el índice está cargado y no vacío.
        
        Returns:
            True si hay documentos indexados, False otherwise.
        """
        return self.index is not None and self.index.ntotal > 0

    def count(self) -> int:
        """
        Retorna el número de vectores en el índice.
        
        Returns:
            Cantidad de chunks indexados.
        """
        return self.index.ntotal if self.index else 0