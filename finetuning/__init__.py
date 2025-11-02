"""
Finetuning module - combines dataset loading and model finetuning.

Available categories:
- code_generation: HumanEval, MBPP, CodeAlpaca, CodeContests
- math_reasoning: GSM8K, MATH (TODO)
- world_knowledge: MMLU, TriviaQA (TODO)
- domain_expertise: MedQA, LegalBench, ArXiv (TODO)
- summarization: CNN/DailyMail, XSum (TODO)
- instruction_following: Alpaca (TODO)
"""

from finetuning.base import BaseDatasetLoader, DatasetSample, DatasetInfo
from finetuning.code_generation import (
    HumanEvalDataset,
    MBPPDataset,
    CodeAlpacaDataset,
    CodeContestsDataset,
    load_code_dataset
)
from finetuning.trainer import ( 
    QuantizedModelFinetuner,
    create_code_format_fn,
    estimate_training_time
)
from finetuning.load_dataset import load_dataset

__all__ = [
    # Base classes
    'BaseDatasetLoader',
    'DatasetSample',
    'DatasetInfo',
    
    # Code generation datasets
    'HumanEvalDataset',
    'MBPPDataset',
    'CodeAlpacaDataset',
    'CodeContestsDataset',
    'load_code_dataset',
    
    # Finetuning
    'QuantizedModelFinetuner',
    'create_code_format_fn',
    'estimate_training_time',
    
    # Factory function
    'load_dataset',
]