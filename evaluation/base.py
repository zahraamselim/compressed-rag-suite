"""Base classes for evaluation benchmarks."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Any, List, Dict
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Type variable for benchmark results
T = TypeVar('T', bound='BenchmarkResult')


@dataclass
class BenchmarkResult:
    """Base class for benchmark results."""
    
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
                        'mfu', 'score', 'map', 'mrr', 'tokens_per_sec']
        
        # Metrics where lower is better
        lower_better = ['latency', 'perplexity', 'loss', 'memory', 'energy', 
                       'ms_per_token', 'time', 'error']
        
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
    def aggregate_from_runs(runs: List['BenchmarkResult'], 
                           metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results from multiple runs with statistics.
        
        Args:
            runs: List of BenchmarkResult instances
            metrics: Specific metrics to aggregate (None = all numeric)
            
        Returns:
            Dict with mean, std, min, max, median for each metric
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
            aggregated[metric] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'n_runs': len(values)
            }
        
        return aggregated
    
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
    
    def _log_metric(self, name: str, value: Any):
        """Log a metric if verbose mode is enabled."""
        if self.verbose:
            if isinstance(value, float):
                logger.info(f"{name}: {value:.4f}")
            else:
                logger.info(f"{name}: {value}")