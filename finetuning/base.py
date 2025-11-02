"""
Base classes for dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Single training/evaluation sample."""
    instruction: str
    input: str
    output: str
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    category: str
    num_train: int
    num_eval: int
    description: str
    source: str
    license: Optional[str] = None


class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    
    Each loader should:
    1. Load data from HuggingFace/local files
    2. Convert to standardized format
    3. Split into train/eval
    4. Provide statistics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.train_data: List[DatasetSample] = []
        self.eval_data: List[DatasetSample] = []
        self._info: Optional[DatasetInfo] = None
    
    @abstractmethod
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """
        Load and return train and eval datasets.
        
        Returns:
            Tuple of (train_samples, eval_samples)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Return dataset metadata."""
        pass
    
    def format_prompt(self, sample: DatasetSample) -> str:
        """
        Format a sample into a prompt string.
        Default format, can be overridden.
        """
        if sample.input:
            return f"""### Instruction:
{sample.instruction}

### Input:
{sample.input}

### Response:
{sample.output}"""
        else:
            return f"""### Instruction:
{sample.instruction}

### Response:
{sample.output}"""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'num_train': len(self.train_data),
            'num_eval': len(self.eval_data),
            'avg_instruction_length': self._avg_length([s.instruction for s in self.train_data]),
            'avg_output_length': self._avg_length([s.output for s in self.train_data]),
        }
    
    def _avg_length(self, texts: List[str]) -> float:
        """Calculate average text length in tokens (approximation)."""
        if not texts:
            return 0.0
        return sum(len(t.split()) for t in texts) / len(texts)