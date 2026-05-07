# Proposal: Buscador Semántico Local (RAG)

## Intent

Implementar un sistema de búsqueda semántica basado en Retrieval-Augmented Generation (RAG) para documentación técnica. El sistema permitirá consultas basadas en intención y contexto, operando de manera local para garantizar privacidad y eficiencia.

## Scope

### In Scope
- Pipeline completo: Ingesta de documentos → Embeddings → Vector Store → Retrieval → Generación de respuesta
- Soporte para archivos PDF, Markdown (.md) y texto (.txt)
- Embeddings locales (Hugging Face) y opción cloud (OpenAI)
- Vector Store local con FAISS
- Motor de búsqueda con similitud del coseno y threshold de 0.7
- CLI interactiva para consultas

### Out of Scope
- Interfaz web/UI (futuro)
- Qdrant o ChromaDB (solo FAISS por ahora)
- Autenticación o multi-usuario
- Integración con otros LLMs localesbeyond OpenAI API

## Capabilities

### New Capabilities
- `document-ingestion`: Sistema de ingesta y procesamiento de documentos
- `semantic-search`: Motor de búsqueda semántica
- `rag-response`: Generación de respuestas con RAG loop
- `vector-persistence`: Almacenamiento vectorial local

### Modified Capabilities
- Ninguno (proyecto nuevo)

## Approach

Arquitectura modular con LangChain:
1. **DocumentLoader** → RecursiveCharacterTextSplitter (chunks 500-1000, overlap 15%)
2. **Embedding** → sentence-transformers (all-MiniLM-L6-v2) o OpenAI
3. **VectorStore** → FAISS con serialización a disco
4. **Retrieval** → Similaridad del coseno con threshold 0.7
5. **RAG Loop** → Context injection + cite sources

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/rag/loader.py` | New | Cargadores de PDF, MD, TXT |
| `src/rag/embedder.py` | New | Generación de embeddings |
| `src/rag/vectorstore.py` | New | Persistencia FAISS |
| `src/rag/search.py` | New | Motor de búsqueda |
| `src/rag/generator.py` | New | RAG loop para respuestas |
| `src/rag/cli.py` | New | Interfaz CLI |
| `data/index/` | New | Índice vectorial persistido |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Embeddings lentos en hardware modesto | Medium | Usar modelo pequeño, cachear |
| PDFs con formato complejo | Low | PyMuPDF loader ya maneja |
| Memoria alta con muchos docs | Medium | Batch processing |

## Rollback Plan

1. Eliminar directorio `data/index/`
2. Re-instalar dependencias si hay conflicto
3. El código fuente queda intacto

## Dependencies

- langchain, langchain-community
- sentence-transformers
- faiss-cpu
- PyMuPDF (para PDFs)
- openai (opcional, para cloud)

## Success Criteria

- [ ] Búsqueda devuelve resultados relevantes sin palabras exactas
- [ ] Persistencia permite reiniciar sin re-indexar
- [ ] Tiempo de respuesta < 2 segundos
- [ ] Threshold 0.7 filtra ruido efectivamente
- [ ] Fuentes citadas en respuesta