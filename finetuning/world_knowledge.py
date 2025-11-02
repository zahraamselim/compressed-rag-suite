"""
World Knowledge Datasets (TODO)

Planned datasets:
1. MMLU - Massive Multitask Language Understanding
2. TriviaQA - Trivia question answering
3. NaturalQuestions - Open domain QA

Implementation needed.
"""

from finetuning.base import BaseDatasetLoader


class MMLUDataset(BaseDatasetLoader):
    """TODO: Implement MMLU dataset loader."""
    def load(self):
        raise NotImplementedError("MMLU dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("MMLU dataset not yet implemented")


class TriviaQADataset(BaseDatasetLoader):
    """TODO: Implement TriviaQA dataset loader."""
    def load(self):
        raise NotImplementedError("TriviaQA dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("TriviaQA dataset not yet implemented")


class NaturalQuestionsDataset(BaseDatasetLoader):
    """TODO: Implement NaturalQuestions dataset loader."""
    def load(self):
        raise NotImplementedError("NaturalQuestions dataset not yet implemented")
    def get_info(self):
        raise NotImplementedError("NaturalQuestions dataset not yet implemented")
