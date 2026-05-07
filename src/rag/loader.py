"""
Cargadores de documentos para el sistema RAG.

Provee funcionalidad para cargar documentos desde múltiples formatos
y convertirlos en fragmentos procesables. Soporta PDF, Markdown y texto plano.

El loader extrae el contenido textual y genera metadatos relevantes
para cada documento (nombre de archivo, tipo, página, fecha de procesamiento).
"""
import os
from pathlib import Path
from datetime import datetime
from typing import List

import fitz
from rag.types import DocumentChunk


class DocumentLoader:
    """
    Cargador de documentos multi-formato.
    
    Maneja la extracción de contenido textual desde PDF, Markdown y archivos
    de texto plano, retornando una lista de DocumentChunk con el contenido
    y metadatos asociados.
    
    Attributes:
        SUPPORTED_FORMATS: Conjunto de extensiones de archivo soportadas.
    
    Example:
        >>> loader = DocumentLoader()
        >>> chunks = loader.load("documento.pdf")
        >>> # Retorna lista de DocumentChunk, uno por página
    """
    
    SUPPORTED_FORMATS = {".pdf", ".md", ".txt"}

    def load(self, path: str) -> List[DocumentChunk]:
        """
        Carga un documento y retorna sus fragmentos.
        
        Detecta el formato del archivo basándose en la extensión y delega
        al loader específico apropiado. Cada formato tiene su propio
        tratamiento:
        
        - PDF: Una página por chunk (para preservar contexto)
        - Markdown: Documento completo como un chunk
        - Texto plano: Documento completo como un chunk
        
        Args:
            path: Ruta al archivo a cargar.
            
        Returns:
            Lista de DocumentChunk con el contenido y metadatos.
            
        Raises:
            ValueError: Si el formato del archivo no es soportado.
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}")

        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".md":
            return self._load_markdown(path)
        elif ext == ".txt":
            return self._load_text(path)

    def _load_pdf(self, path: str) -> List[DocumentChunk]:
        """
        Carga un archivo PDF extrayendo el texto de cada página.
        
        Usa PyMuPDF (fitz) para extraer el contenido textual de cada página.
        Cada página se treata como un chunk independiente para mantener
        el contexto de página en los metadatos.
        
        Args:
            path: Ruta al archivo PDF.
            
        Returns:
            Lista de DocumentChunk, uno por página con contenido válido.
        """
        doc = fitz.open(path)
        chunks = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks.append(DocumentChunk(
                    content=text,
                    metadata={
                        "filename": Path(path).name,
                        "type": "pdf",
                        "page": page_num + 1,
                        "processed_at": datetime.now().isoformat(),
                    }
                ))
        doc.close()
        return chunks

    def _load_markdown(self, path: str) -> List[DocumentChunk]:
        """
        Carga un archivo Markdown retornando el contenido completo.
        
        Args:
            path: Ruta al archivo Markdown.
            
        Returns:
            Lista con un único DocumentChunk conteniendo el texto completo.
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [DocumentChunk(
            content=content,
            metadata={
                "filename": Path(path).name,
                "type": "md",
                "processed_at": datetime.now().isoformat(),
            }
        )]

    def _load_text(self, path: str) -> List[DocumentChunk]:
        """
        Carga un archivo de texto plano retornando el contenido completo.
        
        Args:
            path: Ruta al archivo de texto.
            
        Returns:
            Lista con un único DocumentChunk conteniendo el texto completo.
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [DocumentChunk(
            content=content,
            metadata={
                "filename": Path(path).name,
                "type": "txt",
                "processed_at": datetime.now().isoformat(),
            }
        )]

    def supported_formats(self) -> set:
        """
        Retorna el conjunto de formatos de archivo soportados.
        
        Returns:
            Set con las extensiones soportadas (e.g., {'.pdf', '.md', '.txt'}).
        """
        return self.SUPPORTED_FORMATS.copy()