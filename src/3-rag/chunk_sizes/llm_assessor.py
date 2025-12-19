"""
LLM-based assessment module for RAG chunk quality evaluation.

This module provides functionality to assess the quality of retrieved chunks
using an LLM to evaluate how helpful each chunk is for answering a query.
"""

from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class ChunkAssessment(BaseModel):
    """Assessment of a single chunk"""
    config_name: str = Field(description="Name of the chunking configuration")
    score: int = Field(
        description="Quality score 1-100, where 1 is best and 100 is worst", 
        ge=1, 
        le=100
    )
    reasoning: str = Field(
        description="Brief explanation of the score (1-2 sentences)",
        max_length=1000
    )


class AssessmentResult(BaseModel):
    """Collection of chunk assessments"""
    scores: List[ChunkAssessment] = Field(
        description="List of chunk assessments for each configuration"
    )


def create_llm_client(model_name: str, api_key: str, temperature: float = 0.0):
    """
    Create and configure LLM client for assessment
    
    Args:
        model_name: Model to use (defaults to OPENROUTER_API_MODEL env var)
        temperature: Temperature for generation (0.0 for deterministic)
    
    Returns:
        Configured ChatOpenAI instance
    """    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30,
        max_retries=2,
    )


SYSTEM_PROMPT = """Вы эксперт по оценке качества поиска в RAG (Retrieval-Augmented Generation).

Ваша задача — оценить, насколько полезен каждый фрагмент текста для ответа на запрос пользователя.

ШКАЛА ОЦЕНКИ (1-100):
- 1-20: Очень релевантно - Содержит прямой ответ или ключевую необходимую информацию
- 21-40: Релевантно - Предоставляет полезный контекст или вспомогательную информацию
- 41-60: Частично релевантно - Косвенно связано, может предоставить общую информацию
- 61-80: Минимально релевантно - Едва связано с запросом
- 81-100: Нерелевантно - Бесполезно или не связано с ответом на запрос

ВАЖНЫЕ УКАЗАНИЯ:
1. Ваша задача НЕ отвечать на запрос, а оценивать качество полученной информации
2. Учитывайте: Помогает ли этот фрагмент ответить на запрос? Насколько прямо? Насколько полно?
3. Будьте последовательны в оценке различных фрагментов
4. Предоставьте краткое, конкретное обоснование для каждой оценки
5. Меньшие оценки лучше (1 — лучшая, 100 — худшая)"""

USER_PROMPT = """Запрос пользователя: {query}

Фрагменты текста для оценки:

{chunks}

Оцените каждый фрагмент и предоставьте оценки в указанном формате JSON."""


def build_assessment_prompt():
    """Build the prompt template for LLM assessment"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT + "\n\n{format_instructions}")
    ])


class LLMAssessor:
    """Handles LLM-based assessment of chunk quality"""
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the LLM assessor
        
        Args:
            model_name: LLM model to use for assessment
        """
        self.llm = create_llm_client(model_name, api_key)
        self.parser = PydanticOutputParser(pydantic_object=AssessmentResult)
        self.prompt = build_assessment_prompt()
        self.chain = self.prompt | self.llm | self.parser
    
    def format_chunks_for_prompt(self, chunks_dict: Dict[str, str]) -> str:
        """
        Format chunks dictionary into a string for the prompt
        
        Args:
            chunks_dict: Dict mapping config_name to chunk text
            
        Returns:
            Formatted string with numbered chunks
        """
        formatted = []
        for i, (config_name, chunk_text) in enumerate(chunks_dict.items(), 1):
            formatted.append(f"[{i}] Configuration: {config_name}")
            formatted.append(f"Text: {chunk_text[:500]}...")  # Limit length
            formatted.append("")
        return "\n".join(formatted)
    
    def assess_chunks(self, query: str, chunks_dict: Dict[str, str]) -> AssessmentResult:
        """
        Assess quality of chunks for a given query
        
        Args:
            query: User's query
            chunks_dict: Dict mapping config_name to chunk text
            
        Returns:
            AssessmentResult with scores for each chunk
            
        Raises:
            Exception: If LLM call fails or parsing fails
        """
        chunks_formatted = self.format_chunks_for_prompt(chunks_dict)
        format_instructions = self.parser.get_format_instructions()
        
        result = self.chain.invoke({
            "query": query,
            "chunks": chunks_formatted,
            "format_instructions": format_instructions
        })
        return result
