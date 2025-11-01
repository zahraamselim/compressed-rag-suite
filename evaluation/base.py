"""Base classes for evaluation benchmarks."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Any
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path

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
        """Pretty print results."""
        lines = [f"\n{'='*60}", f"{self.__class__.__name__}", '='*60]
        for key, value in self.to_dict().items():
            if isinstance(value, float):
                lines.append(f"{key:.<40} {value:.4f}")
            else:
                lines.append(f"{key:.<40} {value}")
        lines.append('='*60)
        return '\n'.join(lines)


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
    
    def _log_metric(self, name: str, value: Any):
        """Log a metric if verbose mode is enabled."""
        if self.verbose:
            logger.info(f"{name}: {value}")
