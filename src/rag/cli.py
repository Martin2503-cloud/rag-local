"""
Interfaz CLI y clase principal del sistema RAG.

Este módulo proporciona la clase RAGSystem que orquesta todos los componentes
y un entry point CLI para usar el sistema desde línea de comandos.

Comandos disponibles:
- ingest: Ingesta un documento al índice
- query: Realiza una búsqueda
"""
import sys
from pathlib import Path
from typing import Optional
from openai import OpenAI

from rag.config import load_config
from rag.types import RAGConfig
from rag.loader import DocumentLoader
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.search import SemanticSearch
from rag.generator import RAGGenerator


class RAGSystem:
    """
    Sistema RAG completo que orquesta todos los componentes.
    
    Esta clase integra:
    - DocumentLoader para cargar archivos
    - TextChunker para dividir en chunks
    - Embedder para generar embeddings
    - VectorStore para búsqueda y persistencia
    - SemanticSearch para recuperación
    - OpenAI como LLM opcional
    
    El flujo typical es:
    1. ingest(documento) → carga, chunk, embeber, guardar
    2. load_index() → cargar índice persistido
    3. query(pregunta) → buscar, generar respuesta
    
    Attributes:
        config: Configuración del sistema.
        loader: Cargador de documentos.
        chunker: Divisor de texto.
        embedder: Generador de embeddings.
        vectorstore: Índice vectorial.
        search: Motor de búsqueda.
        llm_client: Cliente de OpenAI (opcional).
        generator: Generador de respuestas.
    
    Example:
        >>> rag = RAGSystem()
        >>> rag.ingest("manual.pdf")
        >>> rag.load_index()
        >>> response = rag.query("cómo funciona el sistema")
        >>> print(response)
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Inicializa el sistema RAG.
        
        Args:
            config: Configuración personalizada (None usa default).
        """
        self.config = config or load_config()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(self.config)
        self.embedder = Embedder(self.config)
        self.vectorstore = VectorStore(self.embedder.dimension)
        self.search = SemanticSearch(self.embedder, self.vectorstore, self.config)
        self.llm_client: Optional[OpenAI] = None
        self.generator = RAGGenerator(self.vectorstore, self.search, self.config)

    def load_index(self) -> None:
        """
        Carga el índice vectorial persistido desde disco.
        
        Busca en la ruta configurada (default: data/index) y carga
        el índice si existe. No falla si no existe.
        """
        if Path(self.config.index_path).exists():
            self.vectorstore.load(self.config.index_path)

    def ingest(self, path: str) -> int:
        """
        Ingesta un documento al sistema.
        
        El pipeline completo:
        1. Cargar documento (PDF/MD/TXT)
        2. Dividir en chunks
        3. Generar embeddings
        4. Agregar al índice
        5. Persistir a disco
        
        Args:
            path: Ruta al archivo a ingestar.
            
        Returns:
            Número de chunks creados.
        """
        docs = self.loader.load(path)
        chunks = self.chunker.chunk_documents(docs)
        chunks = self.embedder.embed_chunks(chunks)
        self.vectorstore.add(chunks)
        self.vectorstore.save(self.config.index_path)
        return len(chunks)

    def query(self, query: str) -> str:
        """
        Procesa una consulta del usuario.
        
        El pipeline:
        1. Embedding de la query
        2. Búsqueda en FAISS
        3. Generación (si hay LLM) o retorno de contexto
        4. Formateo con fuentes
        
        Args:
            query: Pregunta del usuario.
            
        Returns:
            Respuesta formateada con fuentes cited.
        """
        context, results = self.search.search_with_context(query)
        
        if not results:
            return "No tengo información relevante para responder esa pregunta."
        
        if self.llm_client:
            from rag.generator import RAG_PROMPT
            prompt = RAG_PROMPT.format(context=context, question=query)
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            llm_response = response.choices[0].message.content
        else:
            llm_response = context
        
        sources = list(dict.fromkeys(r.source for r in results))
        
        response = llm_response
        if sources:
            response += "\n\nFuentes:\n"
            for i, source in enumerate(sources, 1):
                response += f"- [{i}] {source}\n"
        
        return response


def main():
    """
    Entry point de la CLI.
    
    Uso:
        rag ingest <archivo>  - Ingesta documento
        rag query <pregunta>  - Consulta el sistema
    
    Ejemplos:
        python rag.py ingest docs/manual.pdf
        python rag.py query cómo inicio sesión
    """
    if len(sys.argv) < 2:
        print("Usage: rag <command> [args]")
        print("Commands:")
        print("  ingest <file>   - Ingest a document")
        print("  query <text>   - Query the system")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: rag ingest <file>")
            sys.exit(1)
        file_path = sys.argv[2]
        rag = RAGSystem()
        count = rag.ingest(file_path)
        print(f"Ingested {count} chunks from {file_path}")

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: rag query <text>")
            sys.exit(1)
        query_text = " ".join(sys.argv[2:])
        rag = RAGSystem()
        rag.load_index()
        response = rag.query(query_text)
        print(response)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()