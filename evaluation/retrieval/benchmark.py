"""Retrieval benchmark orchestrator with detailed response saving."""

import logging
import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass

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
    
    # QA-style context quality metrics
    context_sufficiency: Optional[float] = None
    context_precision: Optional[float] = None
    context_coverage: Optional[float] = None
    avg_context_length: Optional[float] = None
    
    # Retrieval consistency metrics
    avg_retrieval_score: Optional[float] = None
    retrieval_consistency: Optional[float] = None
    avg_chunks_retrieved: Optional[float] = None
    
    # Answer quality metrics
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
    evaluation_mode: Optional[str] = None
    num_questions: Optional[int] = None


class RetrievalBenchmark(ModelBenchmark[RetrievalResults]):
    """
    Dual-mode benchmark for RAG retrieval quality.
    
    Mode 1 (IR): Traditional IR metrics (needs relevant_doc_ids)
    Mode 2 (QA): Context quality + answer metrics (needs PDF + QA)
    
    Extended with detailed response saving capabilities.
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
        k_values: Optional[List[int]] = None,
        save_detailed_responses: bool = False,
        output_dir: Optional[str] = None
    ) -> RetrievalResults:
        """
        Run all retrieval benchmarks.
        
        Args:
            questions: List of questions
            ground_truth_answers: Ground truth answers
            relevant_doc_ids: Relevance judgments (for IR metrics)
            documents: Documents to index
            measure_retrieval_quality: Measure IR-style metrics
            measure_context_quality: Measure context quality
            measure_answer_quality: Measure answer quality
            compare_no_rag: Compare with no-RAG baseline
            k_values: K values for precision@k, recall@k
            save_detailed_responses: Save individual Q&A responses
            output_dir: Directory to save detailed responses
            
        Returns:
            RetrievalResults object
        """
        if save_detailed_responses:
            return self._run_with_response_capture(
                questions=questions,
                ground_truth_answers=ground_truth_answers,
                relevant_doc_ids=relevant_doc_ids,
                documents=documents,
                measure_retrieval_quality=measure_retrieval_quality,
                measure_context_quality=measure_context_quality,
                measure_answer_quality=measure_answer_quality,
                compare_no_rag=compare_no_rag,
                k_values=k_values,
                output_dir=output_dir
            )
        else:
            return self._run_standard(
                questions=questions,
                ground_truth_answers=ground_truth_answers,
                relevant_doc_ids=relevant_doc_ids,
                documents=documents,
                measure_retrieval_quality=measure_retrieval_quality,
                measure_context_quality=measure_context_quality,
                measure_answer_quality=measure_answer_quality,
                compare_no_rag=compare_no_rag,
                k_values=k_values
            )
    
    def _run_standard(
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
        """Standard evaluation without detailed response capture."""
        
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
        
        # ALWAYS measure retrieval consistency
        try:
            logger.info("Measuring retrieval consistency metrics...")
            consistency_metrics = self._evaluate_retrieval_consistency(retrieved_contexts)
            for key, value in consistency_metrics.items():
                setattr(results, key, value)
        except Exception as e:
            logger.error(f"Error evaluating retrieval consistency: {e}", exc_info=True)
        
        # IR-style metrics
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
        
        # QA-style context metrics
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
    
    def _run_with_response_capture(
        self,
        questions: List[str],
        ground_truth_answers: Optional[List[str]] = None,
        relevant_doc_ids: Optional[List[Set[str]]] = None,
        documents: Optional[List[str]] = None,
        measure_retrieval_quality: Optional[bool] = None,
        measure_context_quality: Optional[bool] = None,
        measure_answer_quality: Optional[bool] = None,
        compare_no_rag: Optional[bool] = None,
        k_values: Optional[List[int]] = None,
        output_dir: Optional[str] = None
    ) -> RetrievalResults:
        """Evaluation with detailed response capture and saving."""
        
        logger.info(f"Running evaluation with detailed response capture...")
        
        # Use config values
        measure_context_quality = (measure_context_quality 
            if measure_context_quality is not None 
            else self.config.get('measure_context_quality', True))
        measure_answer_quality = (measure_answer_quality 
            if measure_answer_quality is not None 
            else self.config.get('measure_answer_quality', True))
        compare_no_rag = (compare_no_rag 
            if compare_no_rag is not None 
            else self.config.get('compare_no_rag', True))
        
        # Index documents if provided
        if documents:
            logger.info(f"Indexing {len(documents)} documents...")
            self.rag_pipeline.index_documents(documents, show_progress=True)
        
        # Storage for detailed responses
        detailed_results = []
        rag_answers = []
        no_rag_answers = []
        retrieved_contexts_list = []
        
        # Process each question
        for i, question in enumerate(questions):
            ground_truth = ground_truth_answers[i] if ground_truth_answers else ""
            logger.info(f"\nQuestion {i+1}/{len(questions)}: {question[:80]}...")
            
            try:
                # Retrieve contexts
                contexts = self.rag_pipeline.retrieve(question)
                retrieved_contexts_list.append(contexts)
                
                # Generate RAG answer
                rag_answer = self.rag_pipeline.generate_answer(question, contexts)
                rag_answers.append(rag_answer)
                
                # Generate no-RAG answer if needed
                no_rag_answer = ""
                if compare_no_rag:
                    no_rag_answer = self.rag_pipeline.generator.generate_without_context(question)
                no_rag_answers.append(no_rag_answer)
                
                # Compile context
                context_texts = [ctx.get('text', ctx.get('content', '')) for ctx in contexts]
                full_context = '\n\n'.join(context_texts)
                
                # Store detailed result
                detailed_result = {
                    'question_id': i + 1,
                    'question': question,
                    'ground_truth': ground_truth,
                    'rag_answer': rag_answer,
                    'no_rag_answer': no_rag_answer,
                    'num_chunks_retrieved': len(contexts),
                    'context_scores': [ctx.get('score', 0.0) for ctx in contexts],
                    'avg_context_score': sum(ctx.get('score', 0.0) for ctx in contexts) / len(contexts) if contexts else 0.0,
                    'full_context': full_context,
                    'context_length_chars': len(full_context),
                    'rag_answer_length_words': len(rag_answer.split()),
                    'no_rag_answer_length_words': len(no_rag_answer.split()),
                }
                
                detailed_results.append(detailed_result)
                
                logger.info(f"  RAG: {rag_answer[:100]}...")
                if compare_no_rag:
                    logger.info(f"  No-RAG: {no_rag_answer[:100]}...")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                detailed_results.append({
                    'question_id': i + 1,
                    'question': question,
                    'ground_truth': ground_truth,
                    'rag_answer': '',
                    'no_rag_answer': '',
                    'error': str(e)
                })
                rag_answers.append('')
                no_rag_answers.append('')
                retrieved_contexts_list.append([])
        
        # Compute metrics
        logger.info("Computing metrics...")
        
        full_contexts = [
            ' '.join([ctx.get('text', ctx.get('content', '')) for ctx in ctxs])
            for ctxs in retrieved_contexts_list
        ]
        
        # RAG metrics
        metrics = {}
        if ground_truth_answers:
            metrics = self.rag_metrics.evaluate_rag_system(
                questions=questions,
                predictions=rag_answers,
                references=ground_truth_answers,
                contexts=full_contexts,
                predictions_no_rag=no_rag_answers if compare_no_rag else None
            )
        
        # Context quality metrics
        if measure_context_quality and ground_truth_answers:
            context_metrics = self._evaluate_context_quality(
                questions, retrieved_contexts_list, ground_truth_answers
            )
            metrics.update(context_metrics)
        
        # Retrieval consistency
        consistency_metrics = self._evaluate_retrieval_consistency(retrieved_contexts_list)
        metrics.update(consistency_metrics)
        
        # Create results object
        results = RetrievalResults(
            evaluation_mode='qa' if ground_truth_answers else 'retrieval_only',
            num_questions=len(questions)
        )
        
        for key, value in metrics.items():
            if hasattr(results, key):
                setattr(results, key, value)
        
        # Save detailed responses
        if output_dir:
            self._save_detailed_responses(detailed_results, metrics, output_dir)
        
        logger.info("Evaluation with response capture complete!")
        return results
    
    def _save_detailed_responses(
        self,
        detailed_results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        output_dir: str
    ):
        """Save detailed responses in multiple formats."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving detailed responses to {output_dir}...")
        
        # 1. JSON (complete data)
        json_path = output_path / 'detailed_responses.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Saved: {json_path.name}")
        
        # 2. Human-readable text
        text_path = output_path / 'responses_readable.txt'
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED RESPONSES\n")
            f.write("="*80 + "\n\n")
            
            for result in detailed_results:
                f.write(f"\n{'='*80}\n")
                f.write(f"QUESTION {result['question_id']}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Q: {result['question']}\n\n")
                
                if result.get('ground_truth'):
                    f.write(f"GROUND TRUTH:\n{result['ground_truth']}\n\n")
                
                f.write(f"RAG ANSWER ({result.get('rag_answer_length_words', 0)} words):\n")
                f.write(f"{result['rag_answer']}\n\n")
                
                if result.get('no_rag_answer'):
                    f.write(f"NO-RAG ANSWER ({result.get('no_rag_answer_length_words', 0)} words):\n")
                    f.write(f"{result['no_rag_answer']}\n\n")
                
                f.write(f"RETRIEVAL INFO:\n")
                f.write(f"  Chunks: {result.get('num_chunks_retrieved', 0)}\n")
                f.write(f"  Avg Score: {result.get('avg_context_score', 0.0):.4f}\n")
                f.write(f"  Context Length: {result.get('context_length_chars', 0):,} chars\n\n")
                
                if 'context_scores' in result:
                    f.write(f"  Scores: {', '.join([f'{s:.4f}' for s in result['context_scores']])}\n\n")
                
                f.write(f"CONTEXT (first 1000 chars):\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{result.get('full_context', '')[:1000]}...\n")
                f.write(f"{'-'*80}\n")
        
        logger.info(f"  ✓ Saved: {text_path.name}")
        
        # 3. CSV summary
        csv_data = []
        for result in detailed_results:
            csv_data.append({
                'id': result['question_id'],
                'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                'ground_truth': result.get('ground_truth', '')[:100] + '...' if result.get('ground_truth', '') and len(result.get('ground_truth', '')) > 100 else result.get('ground_truth', ''),
                'rag_answer': result['rag_answer'][:100] + '...' if len(result['rag_answer']) > 100 else result['rag_answer'],
                'no_rag_answer': result.get('no_rag_answer', '')[:100] + '...' if result.get('no_rag_answer', '') and len(result.get('no_rag_answer', '')) > 100 else result.get('no_rag_answer', ''),
                'chunks': result.get('num_chunks_retrieved', 0),
                'avg_score': result.get('avg_context_score', 0.0),
                'rag_words': result.get('rag_answer_length_words', 0),
                'no_rag_words': result.get('no_rag_answer_length_words', 0),
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = output_path / 'responses_summary.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ Saved: {csv_path.name}")
        
        # 4. Metrics
        metrics_path = output_path / 'detailed_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  ✓ Saved: {metrics_path.name}")
        
        logger.info(f"All detailed responses saved to {output_dir}")
    
    # Keep existing helper methods unchanged...
    def _evaluate_retrieval_quality(self, questions, retrieved_ids, relevant_doc_ids, k_values):
        """Evaluate retrieval quality using IR metrics."""
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
    
    def _evaluate_retrieval_consistency(self, retrieved_contexts):
        """Evaluate retrieval consistency without ground truth."""
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
        
        if chunk_counts:
            metrics['avg_chunks_retrieved'] = float(np.mean(chunk_counts))
        
        return metrics
    
    def _evaluate_context_quality(self, questions, retrieved_contexts, ground_truth_answers):
        """Evaluate context quality for QA."""
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
            
            full_context = ' '.join([ctx.get('text', ctx.get('content', '')) for ctx in contexts])
            
            if not full_context.strip():
                sufficiency_scores.append(0.0)
                precision_scores.append(0.0)
                coverage_scores.append(0.0)
                context_lengths.append(0)
                continue
            
            # Sufficiency
            sufficiency = self._calculate_context_sufficiency(full_context, answer)
            sufficiency_scores.append(sufficiency)
            
            # Precision
            precision = self.retrieval_metrics.context_relevance(full_context, question)
            precision_scores.append(precision)
            
            # Coverage
            coverage = self._calculate_answer_coverage(full_context, answer)
            coverage_scores.append(coverage)
            
            # Length
            context_lengths.append(len(full_context.split()))
        
        return {
            'context_sufficiency': float(np.mean(sufficiency_scores)) if sufficiency_scores else 0.0,
            'context_precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'context_coverage': float(np.mean(coverage_scores)) if coverage_scores else 0.0,
            'avg_context_length': float(np.mean(context_lengths)) if context_lengths else 0.0
        }
    
    def _calculate_context_sufficiency(self, context: str, answer: str) -> float:
        """Calculate if context contains the answer."""
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        if answer_lower in context_lower:
            return 1.0
        
        threshold = self.config.get('sufficiency_token_threshold', 0.8)
        answer_tokens = set(answer_lower.split())
        context_tokens = set(context_lower.split())
        
        if len(answer_tokens) == 0:
            return 0.0
        
        overlap = len(answer_tokens & context_tokens)
        overlap_ratio = overlap / len(answer_tokens)
        
        return 1.0 if overlap_ratio >= threshold else overlap_ratio
    
    def _calculate_answer_coverage(self, context: str, answer: str) -> float:
        """Calculate fraction of answer tokens in context."""
        context_tokens = set(context.lower().split())
        answer_tokens = set(answer.lower().split())
        
        if len(answer_tokens) == 0:
            return 0.0
        
        covered = len(answer_tokens & context_tokens)
        return covered / len(answer_tokens)
    
    def _generate_no_rag_answers(self, questions: List[str]) -> List[str]:
        """Generate answers without RAG."""
        answers = []
        for question in questions:
            try:
                answer = self.rag_pipeline.generator.generate_without_context(question)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating no-RAG answer: {e}")
                answers.append("")
        return answers
    
    def evaluate_from_file(self, dataset_path: str, **kwargs) -> RetrievalResults:
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
            questions = [item['question'] for item in dataset]
            ground_truth_answers = [item['answer'] for item in dataset if 'answer' in item]
            relevant_doc_ids = None
            documents = None
        elif isinstance(dataset, dict):
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