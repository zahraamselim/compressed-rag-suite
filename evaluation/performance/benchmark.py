"""Performance benchmark orchestrator."""

import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

from evaluation.base import ModelBenchmark, BenchmarkResult
from evaluation.performance.perplexity import PerplexityEvaluator
from evaluation.performance.lm_eval_wrapper import run_lm_eval_harness

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResults(BenchmarkResult):
    """Results from performance benchmarks."""
    perplexity: Optional[float] = None
    lm_eval_scores: Dict[str, float] = field(default_factory=dict)
    average_accuracy: Optional[float] = None
    
    def __str__(self) -> str:
        """Pretty print results with task grouping."""
        lines = [f"\n{'='*60}", f"{self.__class__.__name__}", '='*60]
        
        # Perplexity
        if self.perplexity is not None:
            lines.append(f"{'Perplexity':.<40} {self.perplexity:.2f}")
        
        # LM Eval scores
        if self.lm_eval_scores:
            lines.append(f"\n{'LM-Eval Tasks':.<40}")
            lines.append('-'*60)
            
            # Group tasks by category
            categories = {
                'Reasoning & Common Sense': ['arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 
                                             'siqa', 'winogrande', 'commonsense_qa', 'openbookqa'],
                'Language Understanding': ['lambada', 'storycloze'],
                'Knowledge & QA': ['nq_open', 'triviaqa'],
                'Comprehensive': ['mmlu', 'bbh', 'agieval'],
                'GLUE/SuperGLUE': ['glue', 'super_glue', 'boolq'],
                'Math': ['gsm8k', 'hendrycks_math', 'math_algebra'],
                'Code': ['humaneval', 'mbpp']
            }
            
            for category, task_list in categories.items():
                category_tasks = {k: v for k, v in self.lm_eval_scores.items() if k in task_list}
                if category_tasks:
                    lines.append(f"\n  {category}:")
                    for task, score in sorted(category_tasks.items()):
                        lines.append(f"    {task:.<36} {score:.4f}")
            
            # Uncategorized tasks
            categorized = set(task for tasks in categories.values() for task in tasks)
            other_tasks = {k: v for k, v in self.lm_eval_scores.items() if k not in categorized}
            if other_tasks:
                lines.append(f"\n  Other Tasks:")
                for task, score in sorted(other_tasks.items()):
                    lines.append(f"    {task:.<36} {score:.4f}")
            
            # Average
            if self.average_accuracy is not None:
                lines.append(f"\n{'Average Accuracy':.<40} {self.average_accuracy:.4f}")
        
        lines.append('='*60)
        return '\n'.join(lines)


class PerformanceBenchmark(ModelBenchmark[PerformanceResults]):
    """
    Benchmark suite for measuring model performance/accuracy.
    
    Measures:
        - Perplexity on text datasets
        - Accuracy on standard benchmarks via lm-eval-harness
    """
    
    def __init__(
        self,
        model_interface,
        config: dict,
        verbose: bool = False
    ):
        """
        Initialize performance benchmark.
        
        Args:
            model_interface: ModelInterface instance
            config: Performance config from config.json
            verbose: Enable verbose logging
        """
        super().__init__(
            model_interface=model_interface,
            config=config,
            verbose=verbose
        )
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # Initialize perplexity evaluator
        self.perplexity_evaluator = PerplexityEvaluator(model_interface)
        
        logger.info("Performance benchmark initialized")
    
    def validate_config(self) -> bool:
        """Validate performance configuration."""
        if not super().validate_config():
            return False
        
        # Check if at least one evaluation is enabled
        has_perplexity = self.config.get('measure_perplexity', False)
        has_lm_eval = self.config.get('run_lm_eval', False)
        
        if not has_perplexity and not has_lm_eval:
            logger.warning("No performance metrics enabled in config")
            return False
        
        return True
    
    def run_all(
        self,
        measure_perplexity: Optional[bool] = None,
        run_lm_eval: Optional[bool] = None,
        perplexity_config: Optional[Dict[str, Any]] = None,
        lm_eval_tasks: Optional[Union[List[str], Dict[str, Any]]] = None
    ) -> PerformanceResults:
        """
        Run all performance benchmarks.
        
        Args:
            measure_perplexity: Whether to measure perplexity (uses config if None)
            run_lm_eval: Whether to run lm-eval (uses config if None)
            perplexity_config: Config for perplexity calculation
            lm_eval_tasks: Tasks for lm-eval (uses config if None)
            
        Returns:
            PerformanceResults object with all metrics
        """
        # Use config values if not provided
        measure_perplexity = measure_perplexity if measure_perplexity is not None else self.config.get('measure_perplexity', True)
        run_lm_eval = run_lm_eval if run_lm_eval is not None else self.config.get('run_lm_eval', False)
        
        logger.info("Starting performance benchmarks...")
        
        results = PerformanceResults()
        
        # Measure perplexity
        if measure_perplexity:
            results.perplexity = self._measure_perplexity(perplexity_config)
        
        # Run lm-eval-harness
        if run_lm_eval:
            results.lm_eval_scores = self._run_lm_eval(lm_eval_tasks)
            
            # Calculate average accuracy
            if results.lm_eval_scores:
                results.average_accuracy = sum(results.lm_eval_scores.values()) / len(results.lm_eval_scores)
        
        logger.info("Performance benchmarks complete!")
        if self.verbose:
            print(results)
        
        return results
    
    def _measure_perplexity(
        self,
        perplexity_config: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        Measure perplexity on dataset.
        
        Args:
            perplexity_config: Configuration for perplexity measurement
            
        Returns:
            Perplexity value
        """
        try:
            logger.info("Measuring perplexity...")
            
            # Get config from parameter or self.config
            config = perplexity_config or {}
            
            # Extract perplexity parameters
            dataset_name = config.get('dataset', self.config.get('perplexity_dataset', 'wikitext'))
            dataset_config = config.get('dataset_config', 'wikitext-2-raw-v1')
            split = config.get('split', 'test')
            num_samples = config.get('num_samples', self.config.get('perplexity_num_samples', 100))
            max_length = config.get('max_length', 512)
            stride = config.get('stride', None)
            
            perplexity = self.perplexity_evaluator.calculate(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                num_samples=num_samples,
                max_length=max_length,
                stride=stride
            )
            
            self._log_metric("Perplexity", f"{perplexity:.2f}")
            return perplexity
            
        except Exception as e:
            logger.error(f"Error measuring perplexity: {e}")
            return None
    
    def _run_lm_eval(
        self,
        lm_eval_tasks: Optional[Union[List[str], Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Run lm-evaluation-harness benchmarks.
        
        Args:
            lm_eval_tasks: Either list of task names or dict with task configs
            
        Returns:
            Dictionary of task scores
        """
        try:
            # Check if model supports lm-eval
            if not self.model_interface.supports_lm_eval():
                logger.warning("Model interface does not support lm-eval")
                return {}
            
            logger.info("Running lm-eval-harness...")
            
            # Get tasks from parameter or config
            if lm_eval_tasks is None:
                lm_eval_tasks = self.config.get('lm_eval_tasks', ['hellaswag'])
            
            # Get default parameters (used if not specified per-task)
            default_num_fewshot = self.config.get('lm_eval_num_fewshot', 0)
            default_limit = self.config.get('lm_eval_limit', None)
            default_batch_size = self.config.get('lm_eval_batch_size', 1)
            
            scores = run_lm_eval_harness(
                model_interface=self.model_interface,
                tasks=lm_eval_tasks,
                num_fewshot=default_num_fewshot,
                limit=default_limit,
                batch_size=default_batch_size
            )
            
            # Log individual scores
            for task, score in scores.items():
                self._log_metric(f"lm-eval/{task}", f"{score:.4f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error running lm-eval: {e}")
            return {}
