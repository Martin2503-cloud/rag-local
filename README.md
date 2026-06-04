# rag-local

Buscador semántico local con RAG (Retrieval-Augmented Generation). Sistema modular para búsqueda de documentos usando embeddings locales sin necesidad de API externa.

## Características

- **Ingesta de documentos**: PDF, Markdown, texto plano
- **Embeddings locales**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensiones)
- **Búsqueda semántica**: FAISS con similaridad coseno
- **Generación de respuestas**: Modo sin LLM (retorna contexto) o con OpenAI
- **CLI interactiva**: Comandos para ingestar, buscar y consultar

## Requisitos

- Python 3.12+
- Ver `requirements.txt` para dependencias

## Instalación

```bash
# Clonar el proyecto
git clone https://github.com/Martin2503-cloud/rag-local.git
cd rag-local

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo editable
pip install -e .
```

## Uso

### Ingestar documentos

```bash
rag ingest ./data/documentos
```

### Búsqueda semántica

```bash
rag search "tu consulta aquí"
```

### Consulta con RAG (usando OpenAI)

```bash
rag query "tu pregunta" --llm openai
```

## Configuración

El sistema usa `RAGConfig` con valores por defecto:

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `chunk_size` | 750 | Tamaño de cada chunk en caracteres |
| `chunk_overlap` | 112 | Superposición entre chunks (15%) |
| `embedding_model` | sentence-transformers/all-MiniLM-L6-v2 | Modelo de embeddings |
| `similarity_threshold` | 0.7 | Umbral mínimo de similaridad |
| `top_k` | 4 | Número de resultados a retornar |

Para personalizar, editá los valores en `src/rag/types.py` o pasalos programáticamente.

## Estructura del Proyecto

```
rag-local/
├── src/rag/
│   ├── __init__.py      # Exports públicos
│   ├── config.py       # Carga de configuración
│   ├── loader.py       # Cargadores de documentos
│   ├── chunker.py      # División de texto
│   ├── embedder.py     # Generación de embeddings
│   ├── vectorstore.py  # Persistencia FAISS
│   ├── search.py       # Búsqueda semántica
│   ├── generator.py    # RAG loop
│   ├── cli.py          # Interfaz CLI
│   └── types.py        # Definiciones de tipos
├── tests/              # Suite de tests
├── data/               # Datos e índice persistido
├── requirements.txt    # Dependencias
├── pyproject.toml     # Configuración del paquete
└── README.md
```

## Testing

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=rag --cov-report=html
```

## Dependencias Principales

- **langchain**: Framework para componentes RAG
- **sentence-transformers**: Embeddings locales
- **faiss-cpu**: Búsqueda vectorial
- **pymupdf**: Extracción de PDF
- **openai** (opcional): Para generación de respuestas

## Licencia

MIT