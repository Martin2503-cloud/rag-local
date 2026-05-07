# Design: Buscador Semántico Local (RAG)

## Technical Approach

Sistema RAG modular basado en LangChain con pipeline: Ingesta → Embeddings → FAISS → Retrieval → RAG Loop. Cada capability es un módulo independiente que se comunica vía interfaces definidas.

## Architecture Decisions

### Decision: LangChain over LlamaIndex

**Choice**: LangChain
**Alternatives considered**: LlamaIndex, Custom implementation
**Rationale**: LangChain ofrece más flexibilidad para custom loaders y chain composition. LlamaIndex es más opinionated para RAG fixed.

### Decision: FAISS over ChromaDB

**Choice**: FAISS para persistence local
**Alternatives considered**: ChromaDB, Qdrant
**Rationale**: FAISS es más liviano, no requiere servicio externo, ideal para desarrollo local. ChromaDB tiene más features pero mayor overhead.

### Decision: sentence-transformers local

**Choice**: all-MiniLM-L6-v2 (384 dim)
**Alternatives considered**: OpenAI text-embedding-3-small, Ollama embeddings
**Rationale**: Modelo pequeño (80MB), rápido, no requiere API key. OpenAI como fallback opcional.

### Decision: Configurable chunking

**Choice**: chunk_size=750 (default), overlap=112 (15%)
**Alternatives considered**: Fixed 500, 1000
**Rationale**: 750 es good balance entre contexto y precision. 15% overlap preserva continuity.

### Decision: Cosine Similarity

**Choice**: FAISS with cosIP (inner product) + normalization
**Alternatives considered**: L2 distance
**Rationale**: Cosine es mejor para semantic similarity en embeddings normalizados.

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Document  │────▶│   Loader    │────▶│  Chunker   │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FAISS    │◀────│  VectorDB   │◀────│ Embedder   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                                             
      ▼                                             
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query    │────▶│  Retrieval  │────▶│  Search   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                                             
      ▼                                             
┌─────────────┐     ┌──────────────┐
│   LLM +    │────▶│ RAG Loop   │────▶ Response
│  Context   │     │ (Generator)│     + Sources
└─────────────┘     └──────────────┘
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/rag/__init__.py` | Create | Package init |
| `src/rag/config.py` | Create | Configuración centralizada |
| `src/rag/loader.py` | Create | Document loaders (PDF, MD, TXT) |
| `src/rag/chunker.py` | Create | RecursiveCharacterTextSplitter |
| `src/rag/embedder.py` | Create | Embedding generation |
| `src/rag/vectorstore.py` | Create | FAISS persistence |
| `src/rag/search.py` | Create | Similarity search + threshold |
| `src/rag/generator.py` | Create | RAG loop + prompt |
| `src/rag/cli.py` | Create | CLI interactiva |
| `src/rag/types.py` | Create | Type definitions |
| `data/index/` | Create | Índice persistido |
| `pyproject.toml` | Create | Dependencies |
| `rag.py` | Create | Entry point CLI |

## Interfaces / Contracts

```python
# src/rag/types.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentChunk:
    content: str
    metadata: dict
    embedding: Optional[np.ndarray] = None

@dataclass  
class SearchResult:
    chunk: DocumentChunk
    score: float
    source: str

@dataclass
class RAGConfig:
    chunk_size: int = 750
    chunk_overlap: int = 112
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    top_k: int = 4
```

```python
# src/rag/loader.py
class DocumentLoader:
    def load(self, path: str) -> list[DocumentChunk]: ...
    def supported_formats(self) -> list[str]: ...

# src/rag/embedder.py
class Embedder:
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...

# src/rag/vectorstore.py
class VectorStore:
    def add(self, chunks: list[DocumentChunk]) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def search(self, query_emb: np.ndarray, k: int, threshold: float) -> list[SearchResult]: ...
```

```python
# src/rag/generator.py
class RAGGenerator:
    def __init__(self, vectorstore: VectorStore, llm: Any): ...
    def generate(self, query: str) -> tuple[str, list[str]]: ...
    # Returns: (response, sources)
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | Loader, Chunker, Embedder | Mock files, assert chunks |
| Integration | Full pipeline | Test with sample docs |
| E2E | CLI end-to-end | `rag query "..."` |

### Test Files to Create
- `tests/unit/test_loader.py`
- `tests/unit/test_embedder.py`
- `tests/integration/test_pipeline.py`
- `tests/e2e/test_cli.py`

## Migration / Rollout

No migration required. Proyecto nuevo.

## Open Questions

- [x] Embedding model - all-MiniLM-L6-v2 ✓
- [x] Chunk size - 750 (15% overlap) ✓
- [ ] API key para OpenAI (opcional)
- [ ] LLM provider para generación