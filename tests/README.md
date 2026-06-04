# Documentación de Tests - rag-local

Este documento cubre todas las funciones, clases y fixtures definidos en la suite de tests del proyecto.

## Estructura de Tests

```
tests/
├── test_loader.py      # Tests para DocumentLoader
├── test_chunker.py    # Tests para TextChunker
├── test_embedder.py   # Tests para Embedder
├── test_vectorstore.py # Tests para VectorStore
├── test_search.py     # Tests para SemanticSearch
├── test_generator.py  # Tests para RAGGenerator
└── __init__.py
```

---

## test_loader.py

**Módulo bajo test:** `rag.loader.DocumentLoader`

### Clase: `TestDocumentLoader`

#### Fixtures

| Fixture | Descripción |
|---------|-------------|
| `loader` | Instancia de `DocumentLoader` para usar en todos los tests. |
| `temp_dir` | Directorio temporal que se limpia automáticamente al finalizar. |

#### Tests

| Test | Descripción |
|------|-------------|
| `test_load_markdown` | Verifica que se carga correctamente un archivo `.md` y retorna un chunk con el contenido, tipo `md` y nombre del archivo en metadatos. |
| `test_load_text` | Verifica la carga de archivos de texto plano `.txt`, confirmando contenido y tipo `txt` en metadatos. |
| `test_load_pdf` | Crea un PDF con PyMuPDF (fitz), lo carga y verifica que el contenido, tipo `pdf` y número de página estén en los metadatos. |
| `test_unsupported_format` | Confirma que se lanza `ValueError` con mensaje "Unsupported format" para extensiones no soportadas (e.g., `.xyz`). |
| `test_supported_formats` | Verifica que `supported_formats()` retorne un set que incluye `.pdf`, `.md` y `.txt`. |

---

## test_chunker.py

**Módulo bajo test:** `rag.chunker.TextChunker`

### Clase: `TestTextChunker`

#### Fixtures

| Fixture | Descripción |
|---------|-------------|
| `config` | `RAGConfig` con `chunk_size=50` y `chunk_overlap=10` para tests pequeños. |
| `chunker` | Instancia de `TextChunker` usando la config mencionada. |

#### Tests

| Test | Descripción |
|------|-------------|
| `test_chunk_documents` | Verifica que documentos largos se dividen en más chunks de los originales, y que cada chunk no excede el `chunk_size` (50). |
| `test_chunk_preserves_metadata` | Confirma que al dividir, los metadatos originales (`filename`, `type`) se preservan en los chunks resultantes. |
| `test_chunk_single_document` | Verifica que un documento corto genera al menos un chunk con el contenido intacto. |

---

## test_embedder.py

**Módulo bajo test:** `rag.embedder.Embedder`

### Clase: `TestEmbedder`

#### Fixtures

| Fixture | Descripción |
|---------|-------------|
| `config` | `RAGConfig` especificando `embedding_model="sentence-transformers/all-MiniLM-L6-v2"`. |
| `embedder` | Instancia de `Embedder` con el modelo configurado. |

#### Tests

| Test | Descripción |
|------|-------------|
| `test_embed_dimension` | Verifica que el embedding generado tiene dimensión 384. |
| `test_embed_single_text` | Confirma que un texto produce embeddings de forma `(1, 384)` y está normalizado (norma ~1.0). |
| `test_embed_batch` | Verifica que batch de 3 textos produce array de forma `(3, 384)`. |
| `test_embed_query` | Confirma que `embed_query` retorna un vector de forma `(384,)` normalizado. |
| `test_embed_chunks` | Verifica que `embed_chunks` populate el campo `embedding` de cada `DocumentChunk` y retorna la lista con embeddings. |
| `test_deterministic_embedding` | Confirma que el mismo texto produce exactamente el mismo embedding ( reproducibility). |

---

## test_vectorstore.py

**Módulo bajo test:** `rag.vectorstore.VectorStore`

### Clase: `TestVectorStore`

#### Fixtures

| Fixture | Descripción |
|---------|-------------|
| `store` | `VectorStore` con dimensión 384. |
| `sample_chunks` | Lista de 5 `DocumentChunk` con embeddings aleatorios normalizados. |

#### Tests

| Test | Descripción |
|------|-------------|
| `test_add_chunks` | Verifica que `add()` incrementa el count a 5. |
| `test_search_with_threshold` | Busca con threshold 0.5 y confirma que todos los resultados tienen score >= 0.5. |
| `test_search_returns_ordered_results` | Verifica que los resultados vienen ordenados por score descendente. |
| `test_save_and_load` | Persiste el store, lo recarga y confirma que mantiene los 5 chunks con el contenido original. |
| `test_search_empty_index` | Confirma que buscar en índice vacío lanza `ValueError` con "empty". |
| `test_is_valid` | Verifica que retorna `False` cuando vacío y `True` después de agregar chunks. |
| `test_threshold_filters_results` | Compara resultados con threshold alto (0.9) vs bajo (0.1) y confirma que el alto retorna menos o igual resultados. |

---

## test_search.py

**Módulo bajo test:** `rag.search.SemanticSearch`

### Clase: `TestSemanticSearch`

#### Fixtures

| Fixture | Descripción |
|---------|-------------|
| `config` | `RAGConfig` con `embedding_model`, `top_k=3`, `similarity_threshold=0.5`. |
| `embedder` | Instancia de `Embedder` con la config. |
| `vectorstore` | `VectorStore` con 5 chunks pre-ingestados con embeddings. |
| `search` | `SemanticSearch` con embedder, vectorstore y config. |

#### Tests

| Test | Descripción |
|------|-------------|
| `test_search_returns_results` | Confirma que `search()` retorna una lista no vacía de objetos `SearchResult`. |
| `test_search_with_context` | Verifica que `search_with_context()` retorna una tupla con string de contexto y resultados. |
| `test_search_empty_index` | Confirma que buscar en índice vacío lanza `ValueError`. |
| `test_search_respects_top_k` | Verifica que el número de resultados no excede `top_k` (3). |
| `test_search_threshold_filters` | Confirma que todos los resultados tienen score >= `similarity_threshold` (0.5). |

---

## test_generator.py

**Módulo bajo test:** `rag.generator.RAGGenerator`

### Clase: `TestRAGGenerator`

#### Fixtures

| Fixture | Descripción |
|---------|-------------|
| `config` | `RAGConfig` con `top_k=3`, `similarity_threshold=0.5`. |
| `embedder` | Instancia de `Embedder` con la config. |
| `vectorstore` | `VectorStore` con 3 chunks pre-ingestados. |
| `search` | `SemanticSearch` con embedder, vectorstore y config. |
| `generator_no_llm` | `RAGGenerator` sin cliente LLM (modo debug). |

#### Tests

| Test | Descripción |
|------|-------------|
| `test_generate_without_llm_returns_context` | En modo sin LLM, `generate()` retorna el contexto directamente como string no vacío. |
| `test_generate_returns_sources` | Confirma que retorna una lista de fuentes (puede estar vacía). |
| `test_generate_with_no_results` | Para query sin resultados relevantes, retorna mensaje "No tengo información relevante" y lista vacía de fuentes. |
| `test_format_response_with_sources` | Verifica que formatea respuesta incluyendo las fuentes citadas. |
| `test_format_response_without_sources` | Verifica que formatea respuesta sin lista de fuentes cuando está vacía. |
| `test_generate_with_llm_client` | Mock de OpenAI confirma que se llama al cliente LLM y la respuesta contiene el contenido generado. |

---

## Tipos y Datos Comunes

### `RAGConfig`

Configuración global del sistema RAG con parámetros como:
- `chunk_size`: Tamaño de cada fragmento de texto
- `chunk_overlap`: Superposición entre chunks consecutivos
- `embedding_model`: Modelo de sentence-transformers a usar
- `top_k`: Número de resultados a retornar en búsquedas
- `similarity_threshold`: Threshold mínimo de similitud (0-1)

### `DocumentChunk`

Estructura que representa un fragmento de documento:
- `content`: Texto del chunk
- `metadata`: Diccionario con información adicional (filename, type, page, etc.)
- `embedding`: Vector numpy de 384 dimensiones (populado por el embedder)

### `SearchResult`

Resultado de una búsqueda:
- `chunk`: El `DocumentChunk` encontrado
- `score`: Score de similitud (0-1)
- `source`: Nombre del archivo fuente (del metadata)