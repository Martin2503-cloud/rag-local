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
    
    def __init__(self, config: Optional[RAGConfig] = None, use_llm: bool = False, llm_provider: str = "openai"):
        """
        Inicializa el sistema RAG.
        
        Args:
            config: Configuración personalizada (None usa default).
            use_llm: Si True, inicializa el cliente LLM.
            llm_provider: "openai" o "groq".
        """
        self.config = config or load_config()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(self.config)
        self.embedder = Embedder(self.config)
        self.vectorstore = VectorStore(self.embedder.dimension)
        self.search = SemanticSearch(self.embedder, self.vectorstore, self.config)

        self.llm_client: Optional[OpenAI] = None
        if use_llm:
            self.config.llm_provider = llm_provider
            if llm_provider == "groq" and self.config.groq_api_key:
                try:
                    self.llm_client = OpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=self.config.groq_api_key,
                    )
                    self.config.llm_model = "llama-3.3-70b-versatile"
                except Exception:
                    pass
            elif llm_provider == "openai" and self.config.openai_api_key:
                try:
                    self.llm_client = OpenAI(api_key=self.config.openai_api_key)
                except Exception:
                    pass

        self.generator = RAGGenerator(
            self.vectorstore, self.search, self.config,
            llm_client=self.llm_client,
        )

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
        
        Delega en el RAGGenerator que maneja todo el pipeline:
        búsqueda, generación (con o sin LLM), y formateo con fuentes.
        
        Args:
            query: Pregunta del usuario.
            
        Returns:
            Respuesta formateada con fuentes citadas.
        """
        response, sources = self.generator.generate(query)
        return self.generator.format_response(query, response, sources)


def main():
    """
    Entry point de la CLI.
    
    Uso:
        rag ingest <archivo>                    - Ingesta documento
        rag query <pregunta> [--llm <provider>]  - Consulta el sistema
                                         provider: openai | groq
    
    Ejemplos:
        python rag.py ingest docs/manual.pdf
        python rag.py query cómo inicio sesión
        python rag.py query cómo inicio sesión --llm openai
        python rag.py query cómo inicio sesión --llm groq
    """
    if len(sys.argv) < 2:
        print("Usage: rag <command> [args]")
        print("Commands:")
        print("  ingest <file>                     - Ingest a document")
        print("  query <text> [--llm <provider>]   - Query the system")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: rag ingest <file>")
            sys.exit(1)
        file_path = " ".join(sys.argv[2:])
        rag = RAGSystem()
        count = rag.ingest(file_path)
        print(f"Ingested {count} chunks from {file_path}")

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: rag query <text>")
            sys.exit(1)

        args = sys.argv[2:]
        use_llm = False
        llm_provider = "openai"
        if "--llm" in args:
            idx = args.index("--llm")
            use_llm = True
            args.pop(idx)
            if idx < len(args) and args[idx] in ("openai", "groq"):
                llm_provider = args[idx]
                args.pop(idx)

        query_text = " ".join(args)
        config = load_config()
        if use_llm:
            if llm_provider == "openai" and not config.openai_api_key:
                print("Error: OPENAI_API_KEY no está configurada.")
                print("Configurá la variable de entorno: OPENAI_API_KEY")
                sys.exit(1)
            if llm_provider == "groq" and not config.groq_api_key:
                print("Error: GROQ_API_KEY no está configurada.")
                print("Configurá la variable de entorno: GROQ_API_KEY")
                sys.exit(1)

        rag = RAGSystem(use_llm=use_llm, llm_provider=llm_provider)
        rag.load_index()
        response = rag.query(query_text)
        print(response)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()