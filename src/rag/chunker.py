"""
División de documentos en chunks de texto.

Este módulo contiene la lógica para dividir documentos grandes en fragmentos
manejables de tamaño fijo, usando RecursiveCharacterTextSplitter de LangChain.
El chunking es crítico para la calidad de la búsqueda semántica.
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.types import DocumentChunk, RAGConfig


class TextChunker:
    """
    Divide documentos en chunks de tamaño configurable.
    
    Usa RecursiveCharacterTextSplitter que intenta dividir primero por párrafos,
    luego por líneas, y finalmente por oraciones, preservando la continuidad
    semántica tanto como sea posible.
    
    Attributes:
        config: Configuración con chunk_size y chunk_overlap.
        splitter: Instancia de LangChain para división de texto.
    
    Example:
        >>> config = RAGConfig(chunk_size=750, chunk_overlap=112)
        >>> chunker = TextChunker(config)
        >>> chunks = chunker.chunk_documents(docs)
    """
    
    def __init__(self, config: RAGConfig):
        """
        Inicializa el chunker con la configuración dada.
        
        Args:
            config: RAGConfig con los parámetros de chunking.
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Divide una lista de documentos en chunks pequeños.
        
        Procesa cada documento secuencialmente, dividiendo el texto en fragmentos
        del tamaño especificado en la configuración. Cada chunk resultante
        mantiene los metadatos del documento original más un índice propio.
        
        Args:
            documents: Lista de DocumentChunk a dividir.
            
        Returns:
            Lista de chunks más pequeños con metadatos actualizados.
        """
        chunks = []
        for doc in documents:
            texts = self.splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunks.append(DocumentChunk(
                    content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                    }
                ))
        return chunks