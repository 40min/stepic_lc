"""
Evaluators for RAG chunking quality assessment

This module provides different evaluation strategies for assessing
the quality of chunking configurations.
"""

from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS


class ScoreBasedEvaluator:
    """Evaluates chunks using FAISS distance scores"""
    
    def evaluate(self, dbs: List[FAISS], configs: List[Dict], query: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        Evaluate chunks using FAISS similarity search scores
        
        Args:
            dbs: List of FAISS databases (one per config)
            configs: List of configuration dicts
            query: Query string
            k: Number of chunks to retrieve per config
            
        Returns:
            List of dicts with config, avg_score, scores, and docs_and_scores
        """
        results = []
        
        for i, cfg in enumerate(configs):
            docs_and_scores = dbs[i].similarity_search_with_score(query, k=k)
            scores = [score for _, score in docs_and_scores]
            avg_score = sum(scores) / len(scores) if scores else float('inf')
            
            results.append({
                'config': cfg,
                'avg_score': avg_score,
                'scores': scores,
                'docs_and_scores': docs_and_scores
            })
        
        # Sort by average score (lower is better)
        results.sort(key=lambda x: x['avg_score'])
        return results


class LLMBasedEvaluator:
    """Evaluates chunks using LLM assessment"""
    
    def __init__(self, assessor):
        """
        Initialize LLM evaluator
        
        Args:
            assessor: LLMAssessor instance
        """
        self.assessor = assessor
    
    def evaluate(self, dbs: List[FAISS], configs: List[Dict], query: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        Evaluate chunks using LLM assessment
        
        Args:
            dbs: List of FAISS databases (one per config)
            configs: List of configuration dicts
            query: Query string
            k: Number of chunks to retrieve per config (not used, always retrieves top 1)
            
        Returns:
            List of dicts with config, llm_score, reasoning, and doc
        """
        # Retrieve top chunk from each config
        chunks_dict = {}
        docs_dict = {}
        
        for i, cfg in enumerate(configs):
            docs_and_scores = dbs[i].similarity_search_with_score(query, k=1)
            if docs_and_scores:
                doc, _ = docs_and_scores[0]
                chunks_dict[cfg['name']] = doc.page_content
                docs_dict[cfg['name']] = doc
        
        # Get LLM assessment
        result = self.assessor.assess_chunks(query, chunks_dict)
        
        # Build results list
        results = []
        for assessment in result.scores:
            results.append({
                'config': next(c for c in configs if c['name'] == assessment.config_name),
                'llm_score': assessment.score,
                'reasoning': assessment.reasoning,
                'doc': docs_dict.get(assessment.config_name)
            })
        
        # Sort by LLM score (lower is better)
        results.sort(key=lambda x: x['llm_score'])
        return results
