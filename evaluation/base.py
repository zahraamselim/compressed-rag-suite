"""Base classes for evaluation benchmarks - Enhanced with statistical analysis."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Any, List, Dict, Tuple
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Check for scipy
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.debug("scipy not available for advanced statistics")

# Type variable for benchmark results
T = TypeVar('T', bound='BenchmarkResult')


@dataclass
class BenchmarkResult:
    """Base class for benchmark results with statistical support."""
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save result to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {filepath}")
    
    def __str__(self) -> str:
        """Pretty print results with nested dict support."""
        lines = [f"\n{'='*60}", f"{self.__class__.__name__}", '='*60]
        
        def format_value(v, indent=0):
            """Recursively format values."""
            if isinstance(v, float):
                return f"{v:.4f}"
            elif isinstance(v, dict):
                result = []
                for k, nested_v in v.items():
                    prefix = "  " * (indent + 1)
                    if isinstance(nested_v, dict):
                        result.append(f"{prefix}{k}:")
                        result.append(format_value(nested_v, indent + 1))
                    elif isinstance(nested_v, float):
                        result.append(f"{prefix}{k:.<36} {nested_v:.4f}")
                    else:
                        result.append(f"{prefix}{k:.<36} {nested_v}")
                return '\n'.join(result)
            else:
                return str(v)
        
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                lines.append(format_value(value, 0))
            elif isinstance(value, float):
                lines.append(f"{key:.<40} {value:.4f}")
            else:
                lines.append(f"{key:.<40} {value}")
        
        lines.append('='*60)
        return '\n'.join(lines)
    
    def compare_with(self, other: 'BenchmarkResult', 
                     metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare this result with another.
        
        Args:
            other: Another BenchmarkResult to compare with
            metrics: Specific metrics to compare (None = all numeric metrics)
            
        Returns:
            Dict with comparison details for each metric
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        if metrics is None:
            # Auto-detect numeric metrics
            metrics = [k for k, v in self_dict.items() 
                      if isinstance(v, (int, float))]
        
        comparison = {}
        for metric in metrics:
            if metric not in self_dict or metric not in other_dict:
                logger.warning(f"Metric {metric} not found in one or both results")
                continue
            
            self_val = self_dict[metric]
            other_val = other_dict[metric]
            
            if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                abs_diff = self_val - other_val
                rel_change = (abs_diff / other_val * 100) if other_val != 0 else float('inf')
                
                # Determine if improvement
                is_higher_better = self._is_higher_better(metric)
                improved = (abs_diff > 0) if is_higher_better else (abs_diff < 0)
                
                comparison[metric] = {
                    'baseline': other_val,
                    'current': self_val,
                    'absolute_diff': abs_diff,
                    'relative_change_pct': rel_change,
                    'improved': improved,
                    'higher_is_better': is_higher_better
                }
        
        return comparison
    
    def _is_higher_better(self, metric: str) -> bool:
        """
        Determine if higher values are better for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            True if higher is better, False if lower is better
        """
        # Metrics where higher is better
        higher_better = ['accuracy', 'precision', 'recall', 'f1', 'throughput', 
                        'mfu', 'score', 'map', 'mrr', 'tokens_per_sec', 'speedup',
                        'rouge', 'bleu', 'bertscore', 'exact_match', 'faithfulness']
        
        # Metrics where lower is better
        lower_better = ['latency', 'perplexity', 'loss', 'memory', 'energy', 
                       'ms_per_token', 'time', 'error', 'std']
        
        metric_lower = metric.lower()
        
        # Check if any higher_better keyword is in metric name
        if any(hb in metric_lower for hb in higher_better):
            return True
        
        # Check if any lower_better keyword is in metric name
        if any(lb in metric_lower for lb in lower_better):
            return False
        
        # Default to higher is better
        return True
    
    @staticmethod
    def aggregate_from_runs(
        runs: List['BenchmarkResult'], 
        metrics: Optional[List[str]] = None,
        confidence: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results from multiple runs with statistics and confidence intervals.
        
        Args:
            runs: List of BenchmarkResult instances
            metrics: Specific metrics to aggregate (None = all numeric)
            confidence: Confidence level for intervals (default: 0.95)
            
        Returns:
            Dict with mean, std, std_err, confidence intervals, min, max, median for each metric
        """
        if not runs:
            raise ValueError("No runs provided")
        
        # Collect all metrics across runs
        all_metrics = {}
        for run in runs:
            for key, value in run.to_dict().items():
                if isinstance(value, (int, float)):
                    if metrics is None or key in metrics:
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(float(value))
        
        # Calculate statistics
        aggregated = {}
        for metric, values in all_metrics.items():
            values_array = np.array(values)
            n = len(values)
            mean = np.mean(values_array)
            std = np.std(values_array, ddof=1)  # Sample std
            std_err = std / np.sqrt(n)
            
            stats_dict = {
                'mean': float(mean),
                'std': float(std),
                'std_err': float(std_err),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'n_runs': n
            }
            
            # Add confidence intervals if scipy available
            if SCIPY_AVAILABLE and n > 1:
                t_val = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
                ci_margin = t_val * std_err
                stats_dict['ci_lower'] = float(mean - ci_margin)
                stats_dict['ci_upper'] = float(mean + ci_margin)
                stats_dict['confidence_level'] = confidence
            
            aggregated[metric] = stats_dict
        
        return aggregated
    
    @staticmethod
    def statistical_test(
        baseline_runs: List['BenchmarkResult'],
        comparison_runs: List['BenchmarkResult'],
        metrics: Optional[List[str]] = None,
        alpha: float = 0.05,
        test_type: str = 'ttest'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Test if differences between two sets of runs are statistically significant.
        
        Args:
            baseline_runs: List of baseline BenchmarkResult instances
            comparison_runs: List of comparison BenchmarkResult instances
            metrics: Specific metrics to test (None = all numeric)
            alpha: Significance level (default: 0.05)
            test_type: Type of test ('ttest', 'mannwhitney')
            
        Returns:
            Dict with test results for each metric
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy required for statistical tests")
            return {}
        
        if not baseline_runs or not comparison_runs:
            raise ValueError("Need at least one run in each group")
        
        # Collect metrics
        baseline_metrics = {}
        comparison_metrics = {}
        
        for run in baseline_runs:
            for key, value in run.to_dict().items():
                if isinstance(value, (int, float)):
                    if metrics is None or key in metrics:
                        if key not in baseline_metrics:
                            baseline_metrics[key] = []
                        baseline_metrics[key].append(float(value))
        
        for run in comparison_runs:
            for key, value in run.to_dict().items():
                if isinstance(value, (int, float)):
                    if metrics is None or key in metrics:
                        if key not in comparison_metrics:
                            comparison_metrics[key] = []
                        comparison_metrics[key].append(float(value))
        
        # Perform tests
        results = {}
        for metric in baseline_metrics.keys():
            if metric not in comparison_metrics:
                continue
            
            baseline_vals = np.array(baseline_metrics[metric])
            comparison_vals = np.array(comparison_metrics[metric])
            
            if len(baseline_vals) < 2 or len(comparison_vals) < 2:
                logger.warning(f"Need at least 2 runs per group for {metric}")
                continue
            
            if test_type == 'ttest':
                # Independent samples t-test
                statistic, p_value = scipy_stats.ttest_ind(comparison_vals, baseline_vals)
                test_name = "Independent t-test"
            elif test_type == 'mannwhitney':
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = scipy_stats.mannwhitneyu(
                    comparison_vals, baseline_vals, alternative='two-sided'
                )
                test_name = "Mann-Whitney U test"
            else:
                logger.warning(f"Unknown test type: {test_type}")
                continue
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(baseline_vals) - 1) * np.var(baseline_vals, ddof=1) +
                 (len(comparison_vals) - 1) * np.var(comparison_vals, ddof=1)) /
                (len(baseline_vals) + len(comparison_vals) - 2)
            )
            cohens_d = (np.mean(comparison_vals) - np.mean(baseline_vals)) / pooled_std if pooled_std > 0 else 0.0
            
            results[metric] = {
                'test': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'cohens_d': float(cohens_d),
                'baseline_mean': float(np.mean(baseline_vals)),
                'comparison_mean': float(np.mean(comparison_vals)),
                'baseline_n': len(baseline_vals),
                'comparison_n': len(comparison_vals)
            }
        
        return results
    
    def validate(self) -> bool:
        """
        Validate result data.
        
        Returns:
            True if valid, False otherwise
        """
        result_dict = self.to_dict()
        
        # Check if all values are None
        if all(v is None for v in result_dict.values()):
            logger.error("Result contains only None values")
            return False
        
        # Check for NaN or inf in numeric values
        for key, value in result_dict.items():
            if isinstance(value, float):
                if np.isnan(value):
                    logger.warning(f"Metric '{key}' is NaN")
                    return False
                if np.isinf(value):
                    logger.warning(f"Metric '{key}' is infinite")
                    return False
        
        return True


class ModelBenchmark(ABC, Generic[T]):
    """
    Abstract base class for model benchmarks.
    
    All benchmark classes should inherit from this and implement run_all().
    """
    
    def __init__(
        self,
        model_interface,
        config: dict,
        verbose: bool = False
    ):
        """
        Initialize benchmark.
        
        Args:
            model_interface: ModelInterface instance
            config: Benchmark config from config.json
            verbose: Enable verbose logging
        """
        self.model_interface = model_interface
        self.model = model_interface.get_model()
        self.tokenizer = model_interface.get_tokenizer()
        self.config = config
        self.verbose = verbose
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    @abstractmethod
    def run_all(self, **kwargs) -> T:
        """
        Run all benchmarks and return results.
        
        Returns:
            BenchmarkResult subclass instance
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate benchmark configuration.
        
        Returns:
            True if config is valid
        """
        if self.config is None:
            logger.warning("No configuration provided")
            return False
        return True
    
    def validate_result(self, result: T) -> bool:
        """
        Validate benchmark result before saving.
        
        Args:
            result: BenchmarkResult instance
            
        Returns:
            True if valid
        """
        if result is None:
            logger.error("Result is None")
            return False
        
        return result.validate()
    
    def run_multiple(self, n_runs: int = 5, **kwargs) -> Tuple[T, Dict[str, Dict[str, float]]]:
        """
        Run benchmark multiple times and aggregate results.
        
        Args:
            n_runs: Number of runs to perform
            **kwargs: Arguments to pass to run_all()
            
        Returns:
            Tuple of (mean_result, aggregated_stats)
        """
        logger.info(f"Running benchmark {n_runs} times...")
        
        runs = []
        for i in range(n_runs):
            logger.info(f"Run {i+1}/{n_runs}")
            result = self.run_all(**kwargs)
            if self.validate_result(result):
                runs.append(result)
            else:
                logger.warning(f"Run {i+1} produced invalid results, skipping")
        
        if not runs:
            raise RuntimeError("All runs failed validation")
        
        # Aggregate
        aggregated = BenchmarkResult.aggregate_from_runs(runs)
        
        # Create mean result
        mean_result = runs[0]  # Copy structure from first run
        for key, stats in aggregated.items():
            if hasattr(mean_result, key):
                setattr(mean_result, key, stats['mean'])
        
        logger.info(f"Completed {len(runs)}/{n_runs} successful runs")
        
        return mean_result, aggregated
    
    def _log_metric(self, name: str, value: Any):
        """Log a metric if verbose mode is enabled."""
        if self.verbose:
            if isinstance(value, float):
                logger.info(f"{name}: {value:.4f}")
            else:
                logger.info(f"{name}: {value}")


# Export key components
__all__ = [
    'BenchmarkResult',
    'ModelBenchmark',
    'SCIPY_AVAILABLE'
]
