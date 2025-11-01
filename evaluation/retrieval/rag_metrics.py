"""Metrics for evaluating RAG system performance"""

import logging
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Check numpy availability
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy not available. Install with: pip install numpy")


class RAGMetrics:
    """
    Evaluate end-to-end RAG performance
    - Exact Match, F1 Score
    - ROUGE scores
    - BERTScore
    - Answer quality metrics
    """
    
    def __init__(self):
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for RAG metrics. Install with: pip install numpy")
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for optional metric libraries"""
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.rouge_available = True
        except ImportError:
            self.rouge_available = False
            logger.warning("rouge-score not available. Install: pip install rouge-score")
        
        try:
            from bert_score import score as bert_score
            self.bert_score_fn = bert_score
            self.bertscore_available = True
        except ImportError:
            self.bertscore_available = False
            logger.warning("bert-score not available. Install: pip install bert-score")
    
    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """
        Exact Match: Binary score for exact string match (normalized)
        """
        pred_norm = ' '.join(prediction.lower().strip().split())
        ref_norm = ' '.join(reference.lower().strip().split())
        return float(pred_norm == ref_norm)
    
    @staticmethod
    def token_f1(prediction: str, reference: str) -> float:
        """
        Token-level F1 score
        Measures overlap between prediction and reference tokens
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # Count common tokens
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, Optional[float]]:
        """
        ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        Measures n-gram overlap
        """
        if not self.rouge_available:
            return {'rouge1': None, 'rouge2': None, 'rougeL': None}
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE scoring failed: {e}")
            return {'rouge1': None, 'rouge2': None, 'rougeL': None}
    
    def bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        BERTScore: Semantic similarity using BERT embeddings
        Returns precision, recall, F1
        """
        if not self.bertscore_available or len(predictions) == 0:
            return None
        
        try:
            P, R, F1 = self.bert_score_fn(
                predictions,
                references,
                lang='en',
                verbose=False
            )
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            return None
    
    @staticmethod
    def answer_relevance(answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question
        Based on token overlap
        """
        answer_tokens = set(answer.lower().split())
        question_tokens = set(question.lower().split())
        
        if len(answer_tokens) == 0:
            return 0.0
        
        common = answer_tokens & question_tokens
        return len(common) / len(answer_tokens)
    
    @staticmethod
    def faithfulness_score(answer: str, context: str) -> float:
        """
        Measure how faithful the answer is to the context
        (Does the answer only use information from context?)
        Simple version: token containment
        """
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        if len(answer_tokens) == 0:
            return 0.0
        
        # What fraction of answer tokens appear in context?
        contained = answer_tokens & context_tokens
        return len(contained) / len(answer_tokens)
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str],
        contexts: Optional[List[str]] = None,
        predictions_no_rag: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation
        
        Args:
            questions: List of questions
            predictions: RAG-generated answers
            references: Ground truth answers
            contexts: Retrieved contexts (for faithfulness)
            predictions_no_rag: Answers without RAG (for comparison)
            
        Returns:
            Dict with all metrics
        """
        results = {}
        
        # Basic metrics
        exact_matches = [self.exact_match(pred, ref) 
                        for pred, ref in zip(predictions, references)]
        f1_scores = [self.token_f1(pred, ref) 
                    for pred, ref in zip(predictions, references)]
        relevances = [self.answer_relevance(pred, q) 
                     for pred, q in zip(predictions, questions)]
        
        results['exact_match'] = float(np.mean(exact_matches))
        results['f1_score'] = float(np.mean(f1_scores))
        results['answer_relevance'] = float(np.mean(relevances))
        results['avg_answer_length'] = float(np.mean([len(p.split()) for p in predictions]))
        
        # ROUGE scores
        if self.rouge_available:
            all_rouge = [self.rouge_scores(pred, ref) 
                        for pred, ref in zip(predictions, references)]
            
            # Filter out None values before computing mean
            rouge1_values = [r['rouge1'] for r in all_rouge if r['rouge1'] is not None]
            rouge2_values = [r['rouge2'] for r in all_rouge if r['rouge2'] is not None]
            rougeL_values = [r['rougeL'] for r in all_rouge if r['rougeL'] is not None]
            
            results['rouge1'] = float(np.mean(rouge1_values)) if rouge1_values else None
            results['rouge2'] = float(np.mean(rouge2_values)) if rouge2_values else None
            results['rougeL'] = float(np.mean(rougeL_values)) if rougeL_values else None
        
        # BERTScore
        bert_scores = self.bertscore(predictions, references)
        if bert_scores:
            results['bertscore_precision'] = bert_scores['precision']
            results['bertscore_recall'] = bert_scores['recall']
            results['bertscore_f1'] = bert_scores['f1']
        
        # Faithfulness (if contexts provided)
        if contexts:
            faithfulness_scores = [self.faithfulness_score(pred, ctx) 
                                  for pred, ctx in zip(predictions, contexts)]
            results['faithfulness'] = float(np.mean(faithfulness_scores))
        
        # Comparison with no-RAG (if provided)
        if predictions_no_rag:
            no_rag_f1 = [self.token_f1(pred, ref) 
                        for pred, ref in zip(predictions_no_rag, references)]
            no_rag_em = [self.exact_match(pred, ref) 
                        for pred, ref in zip(predictions_no_rag, references)]
            
            results['no_rag_f1'] = float(np.mean(no_rag_f1))
            results['no_rag_exact_match'] = float(np.mean(no_rag_em))
            results['f1_improvement'] = results['f1_score'] - results['no_rag_f1']
            results['em_improvement'] = results['exact_match'] - results['no_rag_exact_match']
        
        return results
