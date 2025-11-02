"""
Mathematical Reasoning Datasets (TODO)

Planned datasets:
1. GSM8K - Grade school math word problems
2. MATH - Competition mathematics
3. MathQA - Math word problem dataset

Implementation needed.
"""

from finetuning.base import BaseDatasetLoader


class GSM8KDataset(BaseDatasetLoader):
    """TODO: Implement GSM8K dataset loader."""
    def load(self):
        raise NotImplementedError("GSM8K dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("GSM8K dataset not yet implemented")