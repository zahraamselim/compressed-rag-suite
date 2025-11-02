"""
Datasets module for loading and managing training/evaluation datasets.

Available categories:
- code_generation: HumanEval, MBPP, CodeAlpaca, CodeContests
- math_reasoning: GSM8K, MATH (TODO)
- world_knowledge: MMLU, TriviaQA (TODO)
- domain_expertise: MedQA, LegalBench, ArXiv (TODO)
- summarization: CNN/DailyMail, XSum (TODO)
- instruction_following: Alpaca (TODO)
"""

from datasets.base import BaseDatasetLoader, DatasetSample, DatasetInfo
from datasets.code_generation import (
    HumanEvalDataset,
    MBPPDataset,
    CodeAlpacaDataset,
    CodeContestsDataset,
    load_code_dataset
)

__all__ = [
    # Base classes
    'BaseDatasetLoader',
    'DatasetSample',
    'DatasetInfo',
    
    # Code generation
    'HumanEvalDataset',
    'MBPPDataset',
    'CodeAlpacaDataset',
    'CodeContestsDataset',
    'load_code_dataset',
]

def load_dataset(category: str, dataset_name: str, config=None):
    """
    Factory function to load any dataset by category and name.
    
    Args:
        category: Dataset category ('code_generation', 'math_reasoning', etc.)
        dataset_name: Specific dataset name
        config: Dataset configuration
    
    Returns:
        Loaded dataset instance
    
    Example:
        >>> dataset = load_dataset('code_generation', 'mbpp', {'include_tests': True})
        >>> train, eval = dataset.load()
    """
    if category == 'code_generation':
        return load_code_dataset(dataset_name, config)
    else:
        raise NotImplementedError(f"Category '{category}' not yet implemented")
