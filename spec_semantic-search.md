# Semantic Search Specification

## Purpose

Motor de búsqueda semántica que procesa consultas usando embeddings y similitud del coseno.

## Requirements

### Requirement: Embedding Generation

El sistema DEBE transformar texto en vectores densos:

#### Scenario: Generate Embedding from Text

- GIVEN un string de texto
- WHEN se genera el embedding
- THEN el sistema DEBE retornar un vector de dimensiones fixed (384 para all-MiniLM-L6-v2)

#### Scenario: Batch Embedding Generation

- GIVEN múltiples chunks de texto
- WHEN se generan embeddings en batch
- THEN cada chunk DEBE tener su propio vector

### Requirement: Cosine Similarity Search

El sistema DEBE buscar usando similitud del coseno:

#### Scenario: Find Similar Documents

- GIVEN una query y un índice vectorial
- WHEN se ejecuta la búsqueda
- THEN el sistema DEBE retornar documentos ordenados por similarity score

#### Scenario: Apply Score Threshold

- GIVEN threshold de 0.7
- WHEN se ejecuta búsqueda
- THEN DEBE excluir documentos con similarity < 0.7

#### Scenario: Empty Results When No Match

- GIVEN query sin resultados por encima del threshold
- THEN el sistema DEBE retornar lista vacía (no error)

### Requirement: Query Vectorization

El sistema DEBE vectorizar la query del usuario:

#### Scenario: Vectorize User Query

- GIVEN una query del usuario
- WHEN se procesa
- THEN DEBE usar el mismo modelo de embeddings que para indexación

## Error Handling

### Scenario: Empty Index

- GIVEN índice vacío
- WHEN se ejecuta búsqueda
- THEN DEBE indicar que no hay documentos indexados