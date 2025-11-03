"""Metrics for evaluating RAG system performance - Added BLEU score."""

import logging
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy not available. Install with: pip install numpy")


class RAGMetrics:
    """
    Evaluate end-to-end RAG performance.
    All parameters configurable via config.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize RAG metrics with configuration.
        
        Args:
            config: Retrieval evaluation config from config.json
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required for RAG metrics")
        
        self.config = config or {}
        
        # Configurable parameters
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.remove_punctuation = self.config.get('remove_punctuation', False)
        self.faithfulness_method = self.config.get('faithfulness_method', 'token_containment')
        self.relevance_method = self.config.get('relevance_method', 'token_overlap')
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for optional metric libraries."""
        # ROUGE
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=self.config.get('rouge_use_stemmer', True)
            )
            self.rouge_available = True
        except ImportError:
            self.rouge_available = False
            logger.warning("rouge-score not available. Install: pip install rouge-score")
        
        # BERTScore
        try:
            from bert_score import score as bert_score
            self.bert_score_fn = bert_score
            self.bertscore_available = True
            self.bertscore_lang = self.config.get('bertscore_lang', 'en')
            self.bertscore_model = self.config.get('bertscore_model', None)  # Use default
        except ImportError:
            self.bertscore_available = False
            logger.warning("bert-score not available. Install: pip install bert-score")
        
        # BLEU
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer for BLEU...")
                nltk.download('punkt', quiet=True)
            
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.bleu_available = True
            self.smoothing_function = SmoothingFunction().method1
        except ImportError:
            self.bleu_available = False
            logger.warning("NLTK not available for BLEU. Install: pip install nltk")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text based on configuration."""
        if not self.case_sensitive:
            text = text.lower()
        
        if self.normalize_whitespace:
            text = ' '.join(text.strip().split())
        
        if self.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def exact_match(self, prediction: str, reference: str) -> float:
        """
        Exact Match: Binary score for exact string match (normalized).
        """
        pred_norm = self._normalize_text(prediction)
        ref_norm = self._normalize_text(reference)
        return float(pred_norm == ref_norm)
    
    def token_f1(self, prediction: str, reference: str) -> float:
        """
        Token-level F1 score.
        Measures overlap between prediction and reference tokens.
        """
        pred_tokens = self._normalize_text(prediction).split()
        ref_tokens = self._normalize_text(reference).split()
        
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
    
    def bleu_score(self, prediction: str, reference: str) -> Optional[float]:
        """
        BLEU score for translation-style matching.
        Uses sentence-level BLEU with smoothing.
        """
        if not self.bleu_available:
            return None
        
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            # Tokenize
            ref_tokens = [self._normalize_text(reference).split()]
            pred_tokens = self._normalize_text(prediction).split()
            
            if len(pred_tokens) == 0 or len(ref_tokens[0]) == 0:
                return 0.0
            
            # Calculate BLEU with smoothing
            score = sentence_bleu(
                ref_tokens, 
                pred_tokens, 
                smoothing_function=self.smoothing_function
            )
            
            return score
        except Exception as e:
            logger.warning(f"BLEU scoring failed: {e}")
            return None
    
    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, Optional[float]]:
        """
        ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        Measures n-gram overlap.
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
        BERTScore: Semantic similarity using BERT embeddings.
        Returns precision, recall, F1.
        """
        if not self.bertscore_available or len(predictions) == 0:
            return None
        
        try:
            kwargs = {
                'lang': self.bertscore_lang,
                'verbose': False
            }
            if self.bertscore_model:
                kwargs['model_type'] = self.bertscore_model
            
            P, R, F1 = self.bert_score_fn(predictions, references, **kwargs)
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            return None
    
    def answer_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.
        
        Methods:
        - token_overlap: Token overlap ratio
        - semantic: Would use embeddings (not implemented yet)
        """
        if self.relevance_method == 'token_overlap':
            answer_tokens = set(self._normalize_text(answer).split())
            question_tokens = set(self._normalize_text(question).split())
            
            if len(answer_tokens) == 0:
                return 0.0
            
            common = answer_tokens & question_tokens
            return len(common) / len(answer_tokens)
        else:
            logger.warning(f"Unknown relevance method: {self.relevance_method}")
            return 0.0
    
    def faithfulness_score(self, answer: str, context: str) -> float:
        """
        Measure how faithful the answer is to the context.
        
        Methods:
        - token_containment: What fraction of answer tokens appear in context?
        - entailment: Would use NLI model (not implemented yet)
        """
        if self.faithfulness_method == 'token_containment':
            answer_tokens = set(self._normalize_text(answer).split())
            context_tokens = set(self._normalize_text(context).split())
            
            if len(answer_tokens) == 0:
                return 0.0
            
            contained = answer_tokens & context_tokens
            return len(contained) / len(answer_tokens)
        else:
            logger.warning(f"Unknown faithfulness method: {self.faithfulness_method}")
            return 0.0
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str],
        contexts: Optional[List[str]] = None,
        predictions_no_rag: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation.
        
        Args:
            questions: List of questions
            predictions: RAG-generated answers
            references: Ground truth answers
            contexts: Retrieved contexts (for faithfulness)
            predictions_no_rag: Answers without RAG (for comparison)
            
        Returns:
            Dict with all metrics
        """
        if len(questions) != len(predictions) or len(questions) != len(references):
            raise ValueError(f"Length mismatch: questions={len(questions)}, "
                           f"predictions={len(predictions)}, references={len(references)}")
        
        results = {}
        
        # Basic metrics
        exact_matches = [self.exact_match(pred, ref) 
                        for pred, ref in zip(predictions, references)]
        f1_scores = [self.token_f1(pred, ref) 
                    for pred, ref in zip(predictions, references)]
        relevances = [self.answer_relevance(pred, q) 
                     for pred, q in zip(predictions, questions)]
        
        results['exact_match'] = float(np.mean(exact_matches)) if exact_matches else 0.0
        results['f1_score'] = float(np.mean(f1_scores)) if f1_scores else 0.0
        results['answer_relevance'] = float(np.mean(relevances)) if relevances else 0.0
        results['avg_answer_length'] = float(np.mean([len(p.split()) for p in predictions]))
        
        # BLEU scores
        if self.bleu_available:
            bleu_scores = [self.bleu_score(pred, ref) 
                          for pred, ref in zip(predictions, references)]
            bleu_values = [s for s in bleu_scores if s is not None]
            results['bleu'] = float(np.mean(bleu_values)) if bleu_values else None
        
        # ROUGE scores
        if self.rouge_available:
            all_rouge = [self.rouge_scores(pred, ref) 
                        for pred, ref in zip(predictions, references)]
            
            rouge1_values = [r['rouge1'] for r in all_rouge if r['rouge1'] is not None]
            rouge2_values = [r['rouge2'] for r in all_rouge if r['rouge2'] is not None]
            rougeL_values = [r['rougeL'] for r in all_rouge if r['rougeL'] is not None]
            
            results['rouge1'] = float(np.mean(rouge1_values)) if rouge1_values else None
            results['rouge2'] = float(np.mean(rouge2_values)) if rouge2_values else None
            results['rougeL'] = float(np.mean(rougeL_values)) if rougeL_values else None
        
        # BERTScore
        if self.bertscore_available:
            bert_scores = self.bertscore(predictions, references)
            if bert_scores:
                results['bertscore_precision'] = bert_scores['precision']
                results['bertscore_recall'] = bert_scores['recall']
                results['bertscore_f1'] = bert_scores['f1']
        
        # Faithfulness (if contexts provided)
        if contexts:
            if len(contexts) != len(predictions):
                logger.warning(f"Context count ({len(contexts)}) != predictions ({len(predictions)})")
            else:
                faithfulness_scores = [self.faithfulness_score(pred, ctx) 
                                      for pred, ctx in zip(predictions, contexts)]
                results['faithfulness'] = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
        
        # Comparison with no-RAG
        if predictions_no_rag:
            if len(predictions_no_rag) != len(references):
                logger.warning(f"No-RAG predictions count ({len(predictions_no_rag)}) != references ({len(references)})")
            else:
                no_rag_f1 = [self.token_f1(pred, ref) 
                            for pred, ref in zip(predictions_no_rag, references)]
                no_rag_em = [self.exact_match(pred, ref) 
                            for pred, ref in zip(predictions_no_rag, references)]
                
                results['no_rag_f1'] = float(np.mean(no_rag_f1)) if no_rag_f1 else 0.0
                results['no_rag_exact_match'] = float(np.mean(no_rag_em)) if no_rag_em else 0.0
                results['f1_improvement'] = results['f1_score'] - results['no_rag_f1']
                results['em_improvement'] = results['exact_match'] - results['no_rag_exact_match']
        
        return results
