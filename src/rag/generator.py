"""
Generador de respuestas RAG con inyección de contexto.

Este módulo contiene el RAG Loop que combina:
1. Búsqueda semántica para recuperar contexto relevante
2. Inyección del contexto en un prompt estructurado
3. Generación de respuesta (opcional con LLM)
4. Citas de fuentes en la respuesta

El sistema puede operar en dos modos:
- Sin LLM: Retorna los chunks recuperados directamente
- Con OpenAI: Genera respuesta natural usando el contexto
"""
from typing import List, Optional, Any
from openai import OpenAI
from rag.types import SearchResult, RAGConfig
from rag.search import SemanticSearch
from rag.vectorstore import VectorStore


RAG_PROMPT = """Eres un asistente útil. Usa el siguiente contexto para responder la pregunta.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
- Responde basándote ÚNICAMENTE en la información proporcionada en el contexto
- NO inventes información que no esté en el contexto
- Si no tienes suficiente información, dilo claramente
- Cita las fuentes usadas al final de tu respuesta

Respuesta:"""


class RAGGenerator:
    """
    Genera respuestas usando RAG (Retrieval-Augmented Generation).
    
    Orkuesta el pipeline completo de generación:
    1. Busca contexto relevante
    2. Inyecta contexto en prompt
    3. Pide al LLM que genere respuesta
    4. Cita las fuentes
    
    El sistema opera en dos modos:
    - Sin LLM: Retorna el contexto tal cual (útil para debugging)
    - Con OpenAI: Genera respuesta sintetizada
    
    Attributes:
        vectorstore: Índice con documentos.
        search: Motor de búsqueda semántica.
        config: Configuración del sistema.
        llm_client: Cliente de OpenAI (opcional).
    
    Example:
        >>> generator = RAGGenerator(vs, search, config)
        >>> response, sources = generator.generate("qué es RAG?")
        >>> print(response)
    """
    
    def __init__(self, vectorstore: VectorStore, search: SemanticSearch, config: RAGConfig, llm_client: Optional[OpenAI] = None):
        """
        Inicializa el generador.
        
        Args:
            vectorstore: Índice con documentos.
            search: Motor de búsqueda.
            config: Configuración.
            llm_client: Cliente de OpenAI (None para modo sin LLM).
        """
        self.vectorstore = vectorstore
        self.search = search
        self.config = config
        self.llm_client = llm_client

    def generate(self, query: str) -> tuple[str, List[str]]:
        """
        Genera una respuesta para la query del usuario.
        
        El proceso:
        1. Busca chunks relevantes
        2. Si no hay resultados, retorna mensaje de "no encontrado"
        3. Si hay LLM, genera respuesta sintetizada
        4. Si no hay LLM, retorna el contexto directamente
        5. Extrae lista de fuentes únicas
        
        Args:
            query: Pregunta del usuario.
            
        Returns:
            Tupla (respuesta, lista de fuentes).
        """
        context, results = self.search.search_with_context(query)
        
        if not results:
            return "No tengo información relevante para responder esa pregunta.", []
        
        if self.llm_client:
            prompt = RAG_PROMPT.format(context=context, question=query)
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            llm_response = response.choices[0].message.content
        else:
            llm_response = context
        
        sources = list(dict.fromkeys(r.source for r in results))
        
        return llm_response, sources

    def format_response(self, query: str, llm_response: str, sources: List[str]) -> str:
        """
        Formatea la respuesta con las fuentes citadas.
        
        Args:
            llm_response: Respuesta generada por el LLM.
            sources: Lista de archivos fuente.
            
        Returns:
            Respuesta formateada con lista de fuentes.
        """
        response = llm_response
        
        if sources:
            response += "\n\nFuentes:\n"
            for i, source in enumerate(sources, 1):
                response += f"- [{i}] {source}\n"
        
        return response