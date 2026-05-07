# RAG Response Specification

## Purpose

Sistema de generación de respuestas que inyecta contexto recuperado y cita fuentes.

## Requirements

### Requirement: Context Injection

El sistema DEBE inyectar contexto en el prompt:

#### Scenario: Inject Retrieved Context

- GIVEN fragmentos recuperados de la búsqueda
- WHEN se genera la respuesta
- THEN el sistema DEBE incluir los fragmentos en el prompt

#### Scenario: Multiple Relevant Fragments

- GIVEN múltiples fragmentos relevantes (top-k)
- WHEN se genera respuesta
- THEN DEBE usar los k fragmentos más relevantes

### Requirement: Source Citation

El sistema DEBE citar las fuentes:

#### Scenario: Cite Source in Response

- GIVEN fragmentos con metadatos de archivo
- WHEN se genera respuesta
- THEN DEBE incluir referencia al documento fuente
- AND DEBE indicar la fuente en la respuesta generada

#### Scenario: No Sources Available

- GIVEN búsqueda sin resultados relevantes
- WHEN se genera respuesta
- THEN DEBE indicar que no hay información relevante

### Requirement: Hallucination Control

El sistema DEBE prevenir alucinaciones:

#### Scenario: Answer Based on Context Only

- GIVEN contexto retrieved
- WHEN se genera respuesta
- THEN DEBE responder ÚNICAMENTE basándose en el contexto
- AND NO inventar información no presente en el contexto

#### Scenario: Admit Unknown

- GIVEN query sin contexto relevante
- WHEN se genera respuesta
- THEN DEBE indicar que no tiene suficiente información

## Response Format

### Scenario: Formatted Response with Sources

- GIVEN contexto relevante
- WHEN se genera respuesta
- THEN DEBE seguir formato:
  - Respuesta
  - Fuentes:
    - [1] documento1.pdf
    - [2] documento2.pdf