"""Retrieval benchmark orchestrator - FIXED VERSION."""

import logging
import numpy as np
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from pathlib import Path
import json

from evaluation.base import ModelBenchmark, BenchmarkResult
from evaluation.retrieval.retrieval_metrics import RetrievalMetrics
from evaluation.retrieval.rag_metrics import RAGMetrics

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResults(BenchmarkResult):
    """Results from retrieval benchmarks."""
    # Retrieval quality metrics
    precision_at_1: Optional[float] = None
    precision_at_3: Optional[float] = None
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    f1_at_1: Optional[float] = None
    f1_at_3: Optional[float] = None
    f1_at_5: Optional[float] = None
    f1_at_10: Optional[float] = None
    mrr: Optional[float] = None
    map_score: Optional[float] = None
    avg_context_relevance: Optional[float] = None
    
    # RAG end-to-end metrics
    exact_match: Optional[float] = None
    f1_score: Optional[float] = None
    answer_relevance: Optional[float] = None
    faithfulness: Optional[float] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bertscore_f1: Optional[float] = None
    avg_answer_length: Optional[float] = None
    
    # NEW: Retrieval consistency metrics (don't need ground truth!)
    avg_retrieval_score: Optional[float] = None  # Average similarity scores
    retrieval_consistency: Optional[float] = None  # Score variance (lower = more consistent)
    avg_chunks_retrieved: Optional[float] = None  # How many chunks returned
    
    # Comparison metrics
    no_rag_f1: Optional[float] = None
    no_rag_exact_match: Optional[float] = None
    f1_improvement: Optional[float] = None
    em_improvement: Optional[float] = None


class RetrievalBenchmark(ModelBenchmark[RetrievalResults]):
    """
    Benchmark suite for measuring RAG retrieval quality and end-to-end performance.
    """
    
    def __init__(
        self,
        model_interface,
        rag_pipeline,
        config: dict,
        verbose: bool = False
    ):
        super().__init__(
            model_interface=model_interface,
            config=config,
            verbose=verbose
        )
        
        self.rag_pipeline = rag_pipeline
        self.retrieval_metrics = RetrievalMetrics()
        self.rag_metrics = RAGMetrics()
        
        logger.info("Retrieval benchmark initialized")
    
    def validate_config(self) -> bool:
        """Validate retrieval configuration."""
        if not super().validate_config():
            return False
        
        has_retrieval_eval = self.config.get('measure_retrieval_quality', False)
        has_context_eval = self.config.get('measure_context_relevance', False)
        has_answer_eval = self.config.get('measure_answer_faithfulness', False)
        
        if not has_retrieval_eval and not has_context_eval and not has_answer_eval:
            logger.warning("No retrieval metrics enabled in config")
            return False
        
        return True
    
    def run_all(
        self,
        questions: List[str],
        ground_truth_answers: Optional[List[str]] = None,
        relevant_doc_ids: Optional[List[Set[str]]] = None,
        documents: Optional[List[str]] = None,
        measure_retrieval_quality: Optional[bool] = None,
        measure_context_relevance: Optional[bool] = None,
        measure_answer_faithfulness: Optional[bool] = None,
        compare_no_rag: Optional[bool] = None,
        k_values: Optional[List[int]] = None
    ) -> RetrievalResults:
        """Run all retrieval benchmarks."""
        
        if not questions:
            raise ValueError("No questions provided for evaluation")
        
        # Use config values if not provided
        measure_retrieval_quality = (measure_retrieval_quality 
            if measure_retrieval_quality is not None 
            else self.config.get('measure_retrieval_quality', True))
        measure_context_relevance = (measure_context_relevance 
            if measure_context_relevance is not None 
            else self.config.get('measure_context_relevance', True))
        measure_answer_faithfulness = (measure_answer_faithfulness 
            if measure_answer_faithfulness is not None 
            else self.config.get('measure_answer_faithfulness', True))
        compare_no_rag = (compare_no_rag 
            if compare_no_rag is not None 
            else self.config.get('compare_no_rag', True))
        k_values = k_values or self.config.get('k_values', [1, 3, 5, 10])
        
        logger.info(f"Starting retrieval benchmarks on {len(questions)} questions...")
        
        # Index documents if provided
        if documents:
            logger.info(f"Indexing {len(documents)} documents...")
            self.rag_pipeline.index_documents(documents)
        
        results = RetrievalResults()
        
        # Retrieve contexts for all questions
        logger.info("Retrieving contexts for all questions...")
        retrieved_contexts = []
        retrieved_ids = []
        
        for question in questions:
            try:
                contexts = self.rag_pipeline.retrieve(question)
                retrieved_contexts.append(contexts)
                chunk_ids = [ctx.get('chunk_id', ctx.get('id', f'chunk_{i}')) 
                            for i, ctx in enumerate(contexts)]
                retrieved_ids.append(chunk_ids)
            except Exception as e:
                logger.error(f"Error retrieving for question '{question}': {e}")
                retrieved_contexts.append([])
                retrieved_ids.append([])
        
        # FIXED: Measure retrieval quality WITH ground truth
        if measure_retrieval_quality and relevant_doc_ids:
            try:
                logger.info("Measuring retrieval quality with ground truth...")
                retrieval_results = self._evaluate_retrieval_quality(
                    questions, retrieved_ids, relevant_doc_ids, k_values
                )
                for key, value in retrieval_results.items():
                    setattr(results, key.replace('@', '_at_').replace('-', '_'), value)
            except Exception as e:
                logger.error(f"Error evaluating retrieval quality: {e}")
        
        # NEW: Measure retrieval quality WITHOUT ground truth (always useful!)
        if measure_retrieval_quality:
            try:
                logger.info("Measuring retrieval consistency metrics...")
                consistency_metrics = self._evaluate_retrieval_consistency(retrieved_contexts)
                for key, value in consistency_metrics.items():
                    setattr(results, key, value)
            except Exception as e:
                logger.error(f"Error evaluating retrieval consistency: {e}")
        
        # Measure context relevance
        if measure_context_relevance:
            try:
                results.avg_context_relevance = self._evaluate_context_relevance(
                    questions, retrieved_contexts
                )
            except Exception as e:
                logger.error(f"Error evaluating context relevance: {e}")
        
        # Generate answers with RAG
        logger.info("Generating answers with RAG...")
        rag_answers = []
        for question, contexts in zip(questions, retrieved_contexts):
            try:
                answer = self.rag_pipeline.generate_answer(question, contexts)
                rag_answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating answer for question '{question}': {e}")
                rag_answers.append("")
        
        # Measure answer quality and faithfulness
        if ground_truth_answers:
            try:
                full_contexts = [
                    ' '.join([ctx.get('text', ctx.get('content', '')) for ctx in ctxs])
                    for ctxs in retrieved_contexts
                ]
                
                # Generate no-RAG baseline if requested
                no_rag_answers = None
                if compare_no_rag:
                    logger.info("Generating no-RAG baseline answers...")
                    no_rag_answers = self._generate_no_rag_answers(questions)
                
                # Evaluate RAG system
                rag_results = self.rag_metrics.evaluate_rag_system(
                    questions=questions,
                    predictions=rag_answers,
                    references=ground_truth_answers,
                    contexts=full_contexts if measure_answer_faithfulness else None,
                    predictions_no_rag=no_rag_answers
                )
                
                for key, value in rag_results.items():
                    if hasattr(results, key) and value is not None:
                        setattr(results, key, value)
            except Exception as e:
                logger.error(f"Error evaluating answer quality: {e}")
        
        logger.info("Retrieval benchmarks complete!")
        if self.verbose:
            print(results)
        
        return results
    
    def _evaluate_retrieval_quality(
        self,
        questions: List[str],
        retrieved_ids: List[List[str]],
        relevant_doc_ids: List[Set[str]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """Evaluate retrieval quality using IR metrics (needs ground truth)."""
        logger.info("Evaluating retrieval quality...")
        
        retrieval_results = self.retrieval_metrics.evaluate_retrieval(
            queries=questions,
            retrieved_lists=retrieved_ids,
            relevant_sets=relevant_doc_ids,
            k_values=k_values
        )
        
        if 'map' in retrieval_results:
            retrieval_results['map_score'] = retrieval_results.pop('map')
        
        for metric, value in retrieval_results.items():
            self._log_metric(metric, f"{value:.4f}")
        
        return retrieval_results
    
    def _evaluate_retrieval_consistency(
        self,
        retrieved_contexts: List[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        NEW: Evaluate retrieval consistency WITHOUT needing ground truth.
        
        Measures:
        - Average retrieval score (how confident is the system?)
        - Score consistency (do scores vary wildly?)
        - Average chunks retrieved (are we getting results?)
        """
        scores = []
        chunk_counts = []
        
        for contexts in retrieved_contexts:
            if contexts:
                # Extract scores
                context_scores = [ctx.get('score', 0.0) for ctx in contexts]
                scores.extend(context_scores)
                chunk_counts.append(len(contexts))
            else:
                chunk_counts.append(0)
        
        metrics = {}
        
        if scores:
            metrics['avg_retrieval_score'] = float(np.mean(scores))
            metrics['retrieval_consistency'] = float(np.std(scores))  # Lower = more consistent
            self._log_metric("avg_retrieval_score", f"{metrics['avg_retrieval_score']:.4f}")
            self._log_metric("retrieval_consistency", f"{metrics['retrieval_consistency']:.4f}")
        
        if chunk_counts:
            metrics['avg_chunks_retrieved'] = float(np.mean(chunk_counts))
            self._log_metric("avg_chunks_retrieved", f"{metrics['avg_chunks_retrieved']:.2f}")
        
        return metrics
    
    def _evaluate_context_relevance(
        self,
        questions: List[str],
        retrieved_contexts: List[List[Dict[str, Any]]]
    ) -> float:
        """Evaluate how relevant retrieved contexts are to queries."""
        logger.info("Evaluating context relevance...")
        
        relevance_scores = []
        for question, contexts in zip(questions, retrieved_contexts):
            if not contexts:
                relevance_scores.append(0.0)
                continue
            
            full_context = ' '.join([
                ctx.get('text', ctx.get('content', ''))
                for ctx in contexts
            ])
            
            if not full_context.strip():
                relevance_scores.append(0.0)
                continue
            
            relevance = self.retrieval_metrics.context_relevance(full_context, question)
            relevance_scores.append(relevance)
        
        avg_relevance = float(np.mean(relevance_scores)) if relevance_scores else 0.0
        self._log_metric("avg_context_relevance", f"{avg_relevance:.4f}")
        
        return avg_relevance
    
    def _generate_no_rag_answers(self, questions: List[str]) -> List[str]:
        """Generate answers without RAG for comparison."""
        answers = []
        
        for question in questions:
            try:
                answer = self.rag_pipeline.generate_answer(question, contexts=[])
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating no-RAG answer for '{question}': {e}")
                answers.append("")
        
        return answers
    
    def evaluate_from_file(
        self,
        dataset_path: str,
        **kwargs
    ) -> RetrievalResults:
        """Run evaluation from a dataset file."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        questions = dataset.get('questions', [])
        if not questions:
            raise ValueError("Dataset must contain 'questions' field")
        
        ground_truth_answers = dataset.get('ground_truth_answers', None)
        relevant_doc_ids = dataset.get('relevant_doc_ids', None)
        documents = dataset.get('documents', None)
        
        if relevant_doc_ids:
            try:
                relevant_doc_ids = [set(ids) if isinstance(ids, list) else ids 
                                   for ids in relevant_doc_ids]
            except Exception as e:
                logger.error(f"Error converting relevant_doc_ids to sets: {e}")
                relevant_doc_ids = None
        
        logger.info(f"Loaded {len(questions)} questions")
        if ground_truth_answers:
            logger.info(f"Loaded {len(ground_truth_answers)} ground truth answers")
        if relevant_doc_ids:
            logger.info(f"Loaded relevance judgments for {len(relevant_doc_ids)} questions")
        if documents:
            logger.info(f"Loaded {len(documents)} documents")
        
        return self.run_all(
            questions=questions,
            ground_truth_answers=ground_truth_answers,
            relevant_doc_ids=relevant_doc_ids,
            documents=documents,
            **kwargs
        )
