"""Performance benchmark orchestrator - fixed and enhanced."""

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
    perplexity_dataset: Optional[str] = None
    lm_eval_scores: Dict[str, float] = field(default_factory=dict)
    average_accuracy: Optional[float] = None
    num_tasks_evaluated: int = 0
    
    def __str__(self) -> str:
        """Pretty print results with task grouping."""
        lines = [f"\n{'='*60}", f"{self.__class__.__name__}", '='*60]
        
        # Perplexity
        if self.perplexity is not None:
            dataset_info = f" ({self.perplexity_dataset})" if self.perplexity_dataset else ""
            lines.append(f"{'Perplexity' + dataset_info:.<40} {self.perplexity:.4f}")
        
        # LM Eval scores
        if self.lm_eval_scores:
            lines.append(f"\n{'LM-Eval Tasks'} ({self.num_tasks_evaluated} tasks)")
            lines.append('-'*60)
            
            # Group tasks by category
            categories = {
                'Commonsense Reasoning (0-shot)': [
                    'hellaswag', 'winogrande', 'piqa', 'siqa', 
                    'openbookqa', 'arc_easy', 'arc_challenge', 'commonsense_qa'
                ],
                'World Knowledge (5-shot)': [
                    'nq_open', 'triviaqa'
                ],
                'Reading Comprehension (0-shot)': [
                    'boolq', 'quac'
                ],
                'Math': [
                    'gsm8k', 'hendrycks_math', 'math_algebra'
                ],
                'Code': [
                    'humaneval', 'mbpp'
                ],
                'Aggregate Benchmarks': [
                    'mmlu', 'bbh', 'agieval'
                ],
                'Language Understanding': [
                    'lambada', 'storycloze'
                ],
                'NLP Benchmarks': [
                    'glue', 'super_glue'
                ]
            }
            
            for category, task_list in categories.items():
                category_tasks = {k: v for k, v in self.lm_eval_scores.items() if k in task_list}
                if category_tasks:
                    lines.append(f"\n  {category}:")
                    for task, score in sorted(category_tasks.items()):
                        # Convert to percentage for readability
                        score_pct = score * 100
                        lines.append(f"    {task:.<36} {score_pct:>5.2f}%")
            
            # Uncategorized tasks
            categorized = set(task for tasks in categories.values() for task in tasks)
            other_tasks = {k: v for k, v in self.lm_eval_scores.items() if k not in categorized}
            if other_tasks:
                lines.append(f"\n  Other Tasks:")
                for task, score in sorted(other_tasks.items()):
                    score_pct = score * 100
                    lines.append(f"    {task:.<36} {score_pct:>5.2f}%")
            
            # Average
            if self.average_accuracy is not None:
                avg_pct = self.average_accuracy * 100
                lines.append(f"\n{'Average Accuracy':.<40} {avg_pct:>5.2f}%")
        
        lines.append('='*60)
        return '\n'.join(lines)


class PerformanceBenchmark(ModelBenchmark[PerformanceResults]):
    """
    Benchmark suite for measuring model performance/accuracy.
    
    Measures:
        - Perplexity on text datasets (WikiText, etc.)
        - Accuracy on standard benchmarks via lm-eval-harness
    
    Supports the full suite of tasks from the Mistral paper and beyond.
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
        perplexity_config = self.config.get('perplexity', {})
        lm_eval_config = self.config.get('lm_eval', {})
        
        has_perplexity = perplexity_config.get('enabled', False)
        has_lm_eval = lm_eval_config.get('enabled', False)
        
        if not has_perplexity and not has_lm_eval:
            logger.warning("No performance metrics enabled in config")
            # Don't fail validation, just warn
        
        return True
    
    def run_all(
        self,
        measure_perplexity: Optional[bool] = None,
        run_lm_eval: Optional[bool] = None,
        perplexity_kwargs: Optional[Dict[str, Any]] = None,
        lm_eval_kwargs: Optional[Dict[str, Any]] = None
    ) -> PerformanceResults:
        """
        Run all performance benchmarks.
        
        Args:
            measure_perplexity: Whether to measure perplexity (uses config if None)
            run_lm_eval: Whether to run lm-eval (uses config if None)
            perplexity_kwargs: Additional kwargs for perplexity calculation
            lm_eval_kwargs: Additional kwargs for lm-eval
            
        Returns:
            PerformanceResults object with all metrics
        """
        # Validate config
        self.validate_config()
        
        # Determine what to run
        perplexity_config = self.config.get('perplexity', {})
        lm_eval_config = self.config.get('lm_eval', {})
        
        measure_perplexity = (
            measure_perplexity if measure_perplexity is not None 
            else perplexity_config.get('enabled', True)
        )
        run_lm_eval = (
            run_lm_eval if run_lm_eval is not None 
            else lm_eval_config.get('enabled', False)
        )
        
        logger.info("="*60)
        logger.info("Starting performance benchmarks")
        logger.info("="*60)
        
        results = PerformanceResults()
        
        # Measure perplexity
        if measure_perplexity:
            logger.info("\n--- Perplexity Evaluation ---")
            perplexity_result = self._measure_perplexity(
                perplexity_config, 
                perplexity_kwargs
            )
            if perplexity_result:
                results.perplexity = perplexity_result['perplexity']
                results.perplexity_dataset = perplexity_result.get('dataset')
        
        # Run lm-eval-harness
        if run_lm_eval:
            logger.info("\n--- LM-Eval-Harness ---")
            lm_eval_scores = self._run_lm_eval(
                lm_eval_config,
                lm_eval_kwargs
            )
            results.lm_eval_scores = lm_eval_scores
            results.num_tasks_evaluated = len(lm_eval_scores)
            
            # Calculate average accuracy
            if lm_eval_scores:
                results.average_accuracy = sum(lm_eval_scores.values()) / len(lm_eval_scores)
        
        logger.info("\n" + "="*60)
        logger.info("Performance benchmarks complete!")
        logger.info("="*60)
        
        if self.verbose:
            print(results)
        
        return results
    
    def _measure_perplexity(
        self,
        config: Dict[str, Any],
        additional_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Measure perplexity on dataset.
        
        Args:
            config: Perplexity configuration from config.json
            additional_kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with perplexity and metadata
        """
        try:
            # Merge configs
            kwargs = dict(config)
            if additional_kwargs:
                kwargs.update(additional_kwargs)
            
            # Remove 'enabled' flag
            kwargs.pop('enabled', None)
            
            # Extract parameters
            dataset_name = kwargs.pop('dataset', 'wikitext')
            dataset_config = kwargs.pop('dataset_config', 'wikitext-2-raw-v1')
            split = kwargs.pop('split', 'test')
            num_samples = kwargs.pop('num_samples', 100)
            max_length = kwargs.pop('max_length', 512)
            stride = kwargs.pop('stride', None)
            batch_size = kwargs.pop('batch_size', 1)
            
            logger.info(f"Dataset: {dataset_name}/{dataset_config}")
            logger.info(f"Samples: {num_samples}, Max length: {max_length}")
            
            perplexity = self.perplexity_evaluator.calculate(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                num_samples=num_samples,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size
            )
            
            self._log_metric("Perplexity", f"{perplexity:.4f}")
            
            return {
                'perplexity': perplexity,
                'dataset': f"{dataset_name}/{dataset_config}"
            }
            
        except Exception as e:
            logger.error(f"Error measuring perplexity: {e}", exc_info=self.verbose)
            return None
    
    def _run_lm_eval(
        self,
        config: Dict[str, Any],
        additional_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Run lm-evaluation-harness benchmarks.
        
        Args:
            config: LM-Eval configuration from config.json
            additional_kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of task scores
        """
        try:
            # Check if model supports lm-eval
            if not self.model_interface.supports_lm_eval():
                logger.warning("Model interface does not support lm-eval")
                logger.warning("Implement get_lm_eval_model() in your ModelInterface")
                return {}
            
            # Get tasks configuration
            tasks_config = config.get('tasks', {})
            
            # Filter enabled tasks
            enabled_tasks = {}
            for task_name, task_cfg in tasks_config.items():
                if isinstance(task_cfg, dict) and task_cfg.get('enabled', False):
                    enabled_tasks[task_name] = task_cfg
            
            if not enabled_tasks:
                logger.info("No LM-Eval tasks enabled")
                return {}
            
            logger.info(f"Running {len(enabled_tasks)} LM-Eval tasks:")
            for task_name in enabled_tasks.keys():
                logger.info(f"  • {task_name}")
            
            # Get global settings
            global_batch_size = config.get('batch_size', 1)
            
            # Run evaluation
            scores = run_lm_eval_harness(
                model_interface=self.model_interface,
                tasks=enabled_tasks,
                num_fewshot=None,  # Use per-task settings
                limit=None,  # Use per-task settings
                batch_size=global_batch_size
            )
            
            # Log individual scores
            if scores:
                logger.info("\nResults:")
                for task, score in sorted(scores.items()):
                    self._log_metric(f"  {task}", f"{score*100:.2f}%")
                
                avg_score = sum(scores.values()) / len(scores)
                logger.info(f"\nAverage: {avg_score*100:.2f}%")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error running lm-eval: {e}", exc_info=self.verbose)
            return {}