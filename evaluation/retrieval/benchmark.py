"""Retrieval benchmark orchestrator """

import logging
import numpy as np
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import re

from evaluation.base import ModelBenchmark, BenchmarkResult
from evaluation.retrieval.retrieval_metrics import RetrievalMetrics
from evaluation.retrieval.rag_metrics import RAGMetrics

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResults(BenchmarkResult):
    """Results from retrieval benchmarks."""
    # IR-style retrieval metrics (need relevant_doc_ids)
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
    
    # QA-style context quality metrics (don't need relevant_doc_ids!)
    context_sufficiency: Optional[float] = None  # Does context contain answer?
    context_precision: Optional[float] = None  # How much is relevant?
    context_coverage: Optional[float] = None  # % of answer tokens in context
    avg_context_length: Optional[float] = None  # Average context length in tokens
    
    # Retrieval consistency metrics (always available)
    avg_retrieval_score: Optional[float] = None
    retrieval_consistency: Optional[float] = None
    avg_chunks_retrieved: Optional[float] = None
    
    # Answer quality metrics (need ground_truth_answers)
    exact_match: Optional[float] = None
    f1_score: Optional[float] = None
    answer_relevance: Optional[float] = None
    faithfulness: Optional[float] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bertscore_f1: Optional[float] = None
    avg_answer_length: Optional[float] = None
    
    # Comparison metrics
    no_rag_f1: Optional[float] = None
    no_rag_exact_match: Optional[float] = None
    f1_improvement: Optional[float] = None
    em_improvement: Optional[float] = None
    
    # Metadata
    evaluation_mode: Optional[str] = None  # 'ir' or 'qa'
    num_questions: Optional[int] = None


class RetrievalBenchmark(ModelBenchmark[RetrievalResults]):
    """
    Dual-mode benchmark for RAG retrieval quality.
    
    Mode 1 (IR): Traditional IR metrics (needs relevant_doc_ids)
    Mode 2 (QA): Context quality + answer metrics (needs PDF + QA)
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
        self.retrieval_metrics = RetrievalMetrics(config)
        self.rag_metrics = RAGMetrics(config)
        
        logger.info("Retrieval benchmark initialized")
    
    def validate_config(self) -> bool:
        """Validate retrieval configuration."""
        if not super().validate_config():
            return False
        return True
    
    def run_all(
        self,
        questions: List[str],
        ground_truth_answers: Optional[List[str]] = None,
        relevant_doc_ids: Optional[List[Set[str]]] = None,
        documents: Optional[List[str]] = None,
        measure_retrieval_quality: Optional[bool] = None,
        measure_context_quality: Optional[bool] = None,
        measure_answer_quality: Optional[bool] = None,
        compare_no_rag: Optional[bool] = None,
        k_values: Optional[List[int]] = None
    ) -> RetrievalResults:
        """
        Run all retrieval benchmarks in appropriate mode.
        
        Args:
            questions: List of questions
            ground_truth_answers: Ground truth answers (for answer metrics)
            relevant_doc_ids: Relevance judgments (for IR metrics)
            documents: Documents to index (if not already indexed)
            measure_retrieval_quality: Measure IR-style precision/recall
            measure_context_quality: Measure context sufficiency/precision
            measure_answer_quality: Measure answer correctness
            compare_no_rag: Compare with no-RAG baseline
            k_values: K values for precision@k, recall@k
        """
        
        if not questions:
            raise ValueError("No questions provided for evaluation")
        
        # Use config values if not provided
        measure_retrieval_quality = (measure_retrieval_quality 
            if measure_retrieval_quality is not None 
            else self.config.get('measure_retrieval_quality', True))
        measure_context_quality = (measure_context_quality 
            if measure_context_quality is not None 
            else self.config.get('measure_context_quality', True))
        measure_answer_quality = (measure_answer_quality 
            if measure_answer_quality is not None 
            else self.config.get('measure_answer_quality', True))
        compare_no_rag = (compare_no_rag 
            if compare_no_rag is not None 
            else self.config.get('compare_no_rag', True))
        k_values = k_values or self.config.get('k_values', [1, 3, 5, 10])
        
        logger.info(f"Starting retrieval benchmarks on {len(questions)} questions...")
        
        # Determine evaluation mode
        has_relevance_judgments = relevant_doc_ids is not None
        has_ground_truth = ground_truth_answers is not None
        
        if has_relevance_judgments:
            mode = 'ir'
            logger.info("Mode: IR-style evaluation (using relevance judgments)")
        elif has_ground_truth:
            mode = 'qa'
            logger.info("Mode: QA-style evaluation (using ground truth answers)")
        else:
            mode = 'retrieval_only'
            logger.info("Mode: Retrieval consistency only (no ground truth)")
        
        # Index documents if provided
        if documents:
            logger.info(f"Indexing {len(documents)} documents...")
            self.rag_pipeline.index_documents(documents, show_progress=True)
        
        # Check if documents are indexed
        stats = self.rag_pipeline.get_stats()
        if stats['vector_store'].get('count', 0) == 0:
            raise ValueError("No documents indexed! Provide documents or index them first.")
        
        results = RetrievalResults(
            evaluation_mode=mode,
            num_questions=len(questions)
        )
        
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
        
        # ALWAYS measure retrieval consistency (doesn't need ground truth!)
        try:
            logger.info("Measuring retrieval consistency metrics...")
            consistency_metrics = self._evaluate_retrieval_consistency(retrieved_contexts)
            for key, value in consistency_metrics.items():
                setattr(results, key, value)
        except Exception as e:
            logger.error(f"Error evaluating retrieval consistency: {e}", exc_info=True)
        
        # IR-style metrics (need relevance judgments)
        if measure_retrieval_quality and has_relevance_judgments:
            try:
                logger.info("Measuring IR-style retrieval quality...")
                retrieval_results = self._evaluate_retrieval_quality(
                    questions, retrieved_ids, relevant_doc_ids, k_values
                )
                for key, value in retrieval_results.items():
                    setattr(results, key.replace('@', '_at_').replace('-', '_'), value)
            except Exception as e:
                logger.error(f"Error evaluating retrieval quality: {e}", exc_info=True)
        
        # QA-style context metrics (need ground truth answers)
        if measure_context_quality and has_ground_truth:
            try:
                logger.info("Measuring context quality metrics...")
                context_metrics = self._evaluate_context_quality(
                    questions, retrieved_contexts, ground_truth_answers
                )
                for key, value in context_metrics.items():
                    setattr(results, key, value)
            except Exception as e:
                logger.error(f"Error evaluating context quality: {e}", exc_info=True)
        
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
        
        # Measure answer quality
        if measure_answer_quality and has_ground_truth:
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
                    contexts=full_contexts,
                    predictions_no_rag=no_rag_answers
                )
                
                for key, value in rag_results.items():
                    if hasattr(results, key) and value is not None:
                        setattr(results, key, value)
            except Exception as e:
                logger.error(f"Error evaluating answer quality: {e}", exc_info=True)
        
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
        logger.info("Evaluating IR-style retrieval quality...")
        
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
        Evaluate retrieval consistency WITHOUT needing ground truth.
        
        Measures:
        - Average retrieval score
        - Score consistency (std dev)
        - Average chunks retrieved
        """
        scores = []
        chunk_counts = []
        
        for contexts in retrieved_contexts:
            if contexts:
                context_scores = [ctx.get('score', 0.0) for ctx in contexts]
                scores.extend(context_scores)
                chunk_counts.append(len(contexts))
            else:
                chunk_counts.append(0)
        
        metrics = {}
        
        if scores:
            metrics['avg_retrieval_score'] = float(np.mean(scores))
            metrics['retrieval_consistency'] = float(np.std(scores))
            self._log_metric("avg_retrieval_score", f"{metrics['avg_retrieval_score']:.4f}")
            self._log_metric("retrieval_consistency (std)", f"{metrics['retrieval_consistency']:.4f}")
        
        if chunk_counts:
            metrics['avg_chunks_retrieved'] = float(np.mean(chunk_counts))
            self._log_metric("avg_chunks_retrieved", f"{metrics['avg_chunks_retrieved']:.2f}")
        
        return metrics
    
    def _evaluate_context_quality(
        self,
        questions: List[str],
        retrieved_contexts: List[List[Dict[str, Any]]],
        ground_truth_answers: List[str]
    ) -> Dict[str, float]:
        """
        NEW: Evaluate context quality for QA (needs ground truth answers).
        
        Measures:
        - Context sufficiency: Does context contain the answer?
        - Context precision: How much context is relevant?
        - Context coverage: % of answer tokens present in context
        """
        logger.info("Evaluating context quality for QA...")
        
        sufficiency_scores = []
        precision_scores = []
        coverage_scores = []
        context_lengths = []
        
        for question, contexts, answer in zip(questions, retrieved_contexts, ground_truth_answers):
            if not contexts:
                sufficiency_scores.append(0.0)
                precision_scores.append(0.0)
                coverage_scores.append(0.0)
                context_lengths.append(0)
                continue
            
            # Concatenate all retrieved chunks
            full_context = ' '.join([
                ctx.get('text', ctx.get('content', ''))
                for ctx in contexts
            ])
            
            if not full_context.strip():
                sufficiency_scores.append(0.0)
                precision_scores.append(0.0)
                coverage_scores.append(0.0)
                context_lengths.append(0)
                continue
            
            # Context sufficiency: Does context contain answer spans?
            sufficiency = self._calculate_context_sufficiency(full_context, answer)
            sufficiency_scores.append(sufficiency)
            
            # Context precision: Relevance to question
            precision = self.retrieval_metrics.context_relevance(full_context, question)
            precision_scores.append(precision)
            
            # Context coverage: % of answer tokens in context
            coverage = self._calculate_answer_coverage(full_context, answer)
            coverage_scores.append(coverage)
            
            # Context length
            context_lengths.append(len(full_context.split()))
        
        metrics = {
            'context_sufficiency': float(np.mean(sufficiency_scores)) if sufficiency_scores else 0.0,
            'context_precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'context_coverage': float(np.mean(coverage_scores)) if coverage_scores else 0.0,
            'avg_context_length': float(np.mean(context_lengths)) if context_lengths else 0.0
        }
        
        for metric, value in metrics.items():
            self._log_metric(metric, f"{value:.4f}")
        
        return metrics
    
    def _calculate_context_sufficiency(self, context: str, answer: str) -> float:
        """
        Calculate if context contains the answer (fuzzy match).
        
        Checks for:
        1. Exact substring match
        2. Token-level overlap (with configurable threshold)
        """
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        # Exact match
        if answer_lower in context_lower:
            return 1.0
        
        # Token overlap (configurable threshold)
        threshold = self.config.get('sufficiency_token_threshold', 0.8)
        answer_tokens = set(answer_lower.split())
        context_tokens = set(context_lower.split())
        
        if len(answer_tokens) == 0:
            return 0.0
        
        overlap = len(answer_tokens & context_tokens)
        overlap_ratio = overlap / len(answer_tokens)
        
        return 1.0 if overlap_ratio >= threshold else overlap_ratio
    
    def _calculate_answer_coverage(self, context: str, answer: str) -> float:
        """Calculate what fraction of answer tokens appear in context."""
        context_tokens = set(context.lower().split())
        answer_tokens = set(answer.lower().split())
        
        if len(answer_tokens) == 0:
            return 0.0
        
        covered = len(answer_tokens & context_tokens)
        return covered / len(answer_tokens)
    
    def _generate_no_rag_answers(self, questions: List[str]) -> List[str]:
        """Generate answers without RAG for comparison."""
        answers = []
        
        for question in questions:
            try:
                answer = self.rag_pipeline.generator.generate_without_context(question)
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
        
        # Extract questions and answers
        if isinstance(dataset, list):
            # Format: [{"question": ..., "answer": ...}, ...]
            questions = [item['question'] for item in dataset]
            ground_truth_answers = [item['answer'] for item in dataset if 'answer' in item]
            relevant_doc_ids = None
            documents = None
        elif isinstance(dataset, dict):
            # Format: {"questions": [...], "ground_truth_answers": [...], ...}
            questions = dataset.get('questions', [])
            ground_truth_answers = dataset.get('ground_truth_answers', None)
            relevant_doc_ids = dataset.get('relevant_doc_ids', None)
            documents = dataset.get('documents', None)
            
            if relevant_doc_ids:
                relevant_doc_ids = [set(ids) if isinstance(ids, list) else ids 
                                   for ids in relevant_doc_ids]
        else:
            raise ValueError("Dataset must be a list or dict")
        
        if not questions:
            raise ValueError("Dataset must contain 'questions' field")
        
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