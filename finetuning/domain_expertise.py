"""
Domain Expertise Datasets (TODO)

Planned datasets:
1. MedQA - Medical question answering
2. LegalBench - Legal reasoning tasks
3. ArXiv Papers - Scientific domain expertise (custom)

Implementation needed.
"""

from finetuning.base import BaseDatasetLoader


class MedQADataset(BaseDatasetLoader):
    """TODO: Implement MedQA dataset loader."""
    def load(self):
        raise NotImplementedError("MedQA dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("MedQA dataset not yet implemented")


class LegalBenchDataset(BaseDatasetLoader):
    """TODO: Implement LegalBench dataset loader."""
    def load(self):
        raise NotImplementedError("LegalBench dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("LegalBench dataset not yet implemented")


class ArXivDomainDataset(BaseDatasetLoader):
    """TODO: Implement ArXiv domain dataset loader."""
    def load(self):
        raise NotImplementedError("ArXiv domain dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("ArXiv domain dataset not yet implemented")
