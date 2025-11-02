"""
Language/Summarization Datasets (TODO)

Planned datasets:
1. CNN/DailyMail - News summarization
2. XSum - Extreme summarization
3. SAMSum - Dialogue summarization

Implementation needed.
"""

from finetuning.base import BaseDatasetLoader


class CNNDailyMailDataset(BaseDatasetLoader):
    """TODO: Implement CNN/DailyMail dataset loader."""
    def load(self):
        raise NotImplementedError("CNN/DailyMail dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("CNN/DailyMail dataset not yet implemented")


class XSumDataset(BaseDatasetLoader):
    """TODO: Implement XSum dataset loader."""
    def load(self):
        raise NotImplementedError("XSum dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("XSum dataset not yet implemented")


class SAMSumDataset(BaseDatasetLoader):
    """TODO: Implement SAMSum dataset loader."""
    def load(self):
        raise NotImplementedError("SAMSum dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("SAMSum dataset not yet implemented")
