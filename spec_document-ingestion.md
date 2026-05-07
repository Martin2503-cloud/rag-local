# Document Ingestion Specification

## Purpose

Sistema de ingesta y procesamiento de documentos técnicos desde múltiples fuentes (PDF, MD, TXT).

## Requirements

### Requirement: Supported File Formats

El sistema DEBE soportar la carga de documentos en los siguientes formatos:

- PDF (.pdf)
- Markdown (.md)
- Texto plano (.txt)

#### Scenario: Load PDF Document

- GIVEN un archivo PDF válido en la ruta especificada
- WHEN se invoca el loader de documentos
- THEN el sistema DEBE extraer el contenido textual del PDF
- AND DEBE retornar el texto como string

#### Scenario: Load Markdown Document

- GIVEN un archivo Markdown válido
- WHEN se invoca el loader
- THEN DEBE leer y retornar el contenido markdown

#### Scenario: Load Text Document

- GIVEN un archivo .txt válido
- WHEN se invoca el loader
- THEN DEBE retornar el contenido como texto

### Requirement: Text Chunking Strategy

El sistema DEBE dividir el texto en chunks usando RecursiveCharacterTextSplitter:

- Tamaño de chunk: 500-1000 caracteres configurable
- Overlap: 10-15% del tamaño del chunk

#### Scenario: Chunk Within Size Limit

- GIVEN un documento de 2000 caracteres con chunk_size=500
- WHEN se aplica el splitting
- THEN resulting chunks DEBEN tener máximo 500 caracteres
- AND chunks adyacentes DEBEN compartir overlap

#### Scenario: Preserve Context Continuity

- GIVEN texto contínuo con puntuación
- WHEN se divide en chunks
- THEN el sistema DEBE respetar límites de oraciones/párrafos

### Requirement: Document Metadata

Cada documento DEBE incluir metadatos:

- Nombre del archivo
- Tipo de archivo
- Fecha de procesamiento

#### Scenario: Extract Metadata

- GIVEN un archivo "manual.pdf"
- WHEN se procesa el documento
- THEN DEBE incluir metadata con {filename: "manual.pdf", type: "pdf"}

## Error Handling

### Scenario: Invalid File Format

- GIVEN un archivo con extensión no soportada (.docx)
- WHEN se intenta cargar
- THEN el sistema DEBE lanzar un error indicando formato no soportado

### Scenario: Corrupted File

- GIVEN un archivo PDF corrupto
- WHEN se intenta cargar
- THEN el sistema DEBE manejar el error gracefully y NO fallar