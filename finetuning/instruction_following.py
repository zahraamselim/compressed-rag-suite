"""
Instruction Following Datasets (TODO)

Planned datasets:
1. Alpaca - Instruction following
2. Dolly - Instruction dataset
3. FLAN - Instruction tuning

Implementation needed.
"""

from finetuning.base import BaseDatasetLoader


class AlpacaDataset(BaseDatasetLoader):
    """TODO: Implement Alpaca dataset loader."""
    def load(self):
        raise NotImplementedError("Alpaca dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("Alpaca dataset not yet implemented")