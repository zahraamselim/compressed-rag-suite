"""
Evaluation utilities for finetuned models.

Provides task-specific evaluation metrics:
- Code generation: pass@k, execution accuracy
- Math reasoning: exact match, numerical accuracy
- General: BLEU, ROUGE, perplexity
"""

import logging
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Results from model evaluation."""
    task: str
    metrics: Dict[str, float]
    num_samples: int
    predictions: List[str] = None
    references: List[str] = None
    
    def to_dict(self) -> dict:
        return {
            'task': self.task,
            'metrics': self.metrics,
            'num_samples': self.num_samples
        }


class CodeGenerationEvaluator:
    """
    Evaluator for code generation tasks.
    
    Metrics:
    - Syntax correctness (can parse)
    - Execution accuracy (passes tests)
    - pass@k (probability of k samples passing)
    """
    
    def __init__(self):
        self.task = 'code_generation'
    
    def evaluate(
        self,
        predictions: List[str],
        test_cases: List[List[str]],
        entry_points: List[str]
    ) -> EvaluationResults:
        """
        Evaluate code generation predictions.
        
        Args:
            predictions: Generated code samples
            test_cases: Test cases for each problem
            entry_points: Function names to test
        
        Returns:
            EvaluationResults with pass@1, syntax correctness, etc.
        """
        if len(predictions) != len(test_cases):
            raise ValueError("Predictions and test_cases must have same length")
        
        syntax_correct = []
        execution_correct = []
        
        for pred, tests, entry in zip(predictions, test_cases, entry_points):
            # Check syntax
            try:
                compile(pred, '<string>', 'exec')
                syntax_correct.append(1.0)
            except SyntaxError:
                syntax_correct.append(0.0)
                execution_correct.append(0.0)
                continue
            
            # Check execution (simplified - real implementation needs sandboxing)
            try:
                # Execute code and tests
                namespace = {}
                exec(pred, namespace)
                
                all_passed = True
                for test in tests:
                    try:
                        exec(test, namespace)
                    except:
                        all_passed = False
                        break
                
                execution_correct.append(1.0 if all_passed else 0.0)
            except:
                execution_correct.append(0.0)
        
        metrics = {
            'syntax_correctness': float(np.mean(syntax_correct)),
            'pass@1': float(np.mean(execution_correct)),
            'num_correct': int(sum(execution_correct)),
        }
        
        return EvaluationResults(
            task=self.task,
            metrics=metrics,
            num_samples=len(predictions),
            predictions=predictions
        )


class TextGenerationEvaluator:
    """
    General text generation evaluator.
    
    Metrics:
    - BLEU score
    - ROUGE scores
    - Exact match
    - Token F1
    """
    
    def __init__(self):
        self.task = 'text_generation'
        
        # Try to import optional dependencies
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.rouge_available = True
        except ImportError:
            self.rouge_available = False
            logger.warning("rouge-score not available")
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> EvaluationResults:
        """
        Evaluate text generation predictions.
        
        Args:
            predictions: Generated text samples
            references: Reference answers
        
        Returns:
            EvaluationResults with ROUGE, exact match, etc.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        metrics = {}
        
        # Exact match
        exact_matches = [
            float(pred.strip().lower() == ref.strip().lower())
            for pred, ref in zip(predictions, references)
        ]
        metrics['exact_match'] = float(np.mean(exact_matches))
        
        # Token F1
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            
            common = pred_tokens & ref_tokens
            if not common:
                f1_scores.append(0.0)
                continue
            
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        
        metrics['token_f1'] = float(np.mean(f1_scores))
        
        # ROUGE scores
        if self.rouge_available:
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            metrics['rouge1'] = float(np.mean(rouge1_scores))
            metrics['rouge2'] = float(np.mean(rouge2_scores))
            metrics['rougeL'] = float(np.mean(rougeL_scores))
        
        return EvaluationResults(
            task=self.task,
            metrics=metrics,
            num_samples=len(predictions),
            predictions=predictions,
            references=references
        )


def create_evaluator(task: str):
    """
    Factory function to create appropriate evaluator.
    
    Args:
        task: Task type ('code_generation', 'text_generation', etc.)
    
    Returns:
        Evaluator instance
    """
    evaluators = {
        'code_generation': CodeGenerationEvaluator,
        'text_generation': TextGenerationEvaluator,
    }
    
    if task not in evaluators:
        raise ValueError(f"Unknown task: {task}. Available: {list(evaluators.keys())}")
    
    return evaluators[task]()