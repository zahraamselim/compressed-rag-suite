"""Metrics for evaluating retrieval quality"""

import numpy as np
from typing import List, Dict, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Evaluate retrieval quality with standard IR metrics.
    """
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Precision@K: What fraction of retrieved items are relevant?
        
        Args:
            retrieved: List of retrieved chunk IDs
            relevant: Set of relevant chunk IDs
            k: Cut-off point
        """
        if k == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_count = sum(1 for chunk_id in retrieved_k if chunk_id in relevant)
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Recall@K: What fraction of relevant items are retrieved?
        """
        if len(relevant) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_count = sum(1 for chunk_id in retrieved_k if chunk_id in relevant)
        return relevant_count / len(relevant)
    
    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """F1@K: Harmonic mean of Precision@K and Recall@K"""
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
        """
        MRR: 1 / rank of first relevant item
        Measures how quickly we find a relevant item
        """
        for i, chunk_id in enumerate(retrieved, 1):
            if chunk_id in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Average Precision: Average of precision values at each relevant item
        """
        if len(relevant) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0
        
        for i, chunk_id in enumerate(retrieved, 1):
            if chunk_id in relevant:
                num_hits += 1
                score += num_hits / i
        
        return score / len(relevant)
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant_scores: Dict[str, float], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Considers graded relevance (not just binary)
        
        Args:
            retrieved: List of retrieved chunk IDs
            relevant_scores: Dict mapping chunk_id to relevance score (0-1 or 0-5)
            k: Cut-off point
        """
        if k == 0:
            return 0.0
            
        retrieved_k = retrieved[:k]
        
        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved_k, 1):
            rel = relevant_scores.get(chunk_id, 0.0)
            dcg += rel / np.log2(i + 1)
        
        # IDCG: Ideal DCG (if we ranked perfectly)
        ideal_scores = sorted(relevant_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def context_relevance(retrieved_text: str, query: str) -> float:
        """
        Simple relevance score based on token overlap
        """
        query_tokens = set(query.lower().split())
        context_tokens = set(retrieved_text.lower().split())
        
        if len(query_tokens) == 0:
            return 0.0
        
        overlap = len(query_tokens & context_tokens)
        return overlap / len(query_tokens)
    
    @staticmethod
    def evaluate_retrieval(
        queries: List[str],
        retrieved_lists: List[List[str]],
        relevant_sets: List[Set[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval evaluation
        
        Args:
            queries: List of query strings
            retrieved_lists: List of retrieved chunk ID lists (one per query)
            relevant_sets: List of relevant chunk ID sets (one per query)
            k_values: K values to evaluate at
            
        Returns:
            Dict with all metrics averaged across queries
        """
        results = {f"precision@{k}": [] for k in k_values}
        results.update({f"recall@{k}": [] for k in k_values})
        results.update({f"f1@{k}": [] for k in k_values})
        results["mrr"] = []
        results["map"] = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            # Precision, Recall, F1 at different K
            for k in k_values:
                results[f"precision@{k}"].append(
                    RetrievalMetrics.precision_at_k(retrieved, relevant, k)
                )
                results[f"recall@{k}"].append(
                    RetrievalMetrics.recall_at_k(retrieved, relevant, k)
                )
                results[f"f1@{k}"].append(
                    RetrievalMetrics.f1_at_k(retrieved, relevant, k)
                )
            
            # MRR and MAP
            results["mrr"].append(
                RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant)
            )
            results["map"].append(
                RetrievalMetrics.average_precision(retrieved, relevant)
            )
        
        # Average all metrics
        return {metric: np.mean(values) for metric, values in results.items()}
