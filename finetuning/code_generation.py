"""
Code Generation Datasets

Available datasets:
1. HumanEval - Python code generation (pass@k evaluation)
2. MBPP - Mostly Basic Python Problems
3. CodeAlpaca - Instruction-tuning for code
4. CodeContests - Competitive programming
"""

from datasets import load_dataset as hf_load_dataset  # Fixed: use alias
from typing import List, Tuple, Optional, Dict, Any
import random
import logging

from finetuning.base import BaseDatasetLoader, DatasetSample, DatasetInfo

logger = logging.getLogger(__name__)


class HumanEvalDataset(BaseDatasetLoader):
    """
    HumanEval dataset for Python code generation.
    
    Source: https://github.com/openai/human-eval
    Format: Function signature + docstring → Implementation
    Evaluation: pass@k metric (k=1,10,100)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.eval_only = config.get('eval_only', True) if config else True
        self.include_tests = config.get('include_tests', False) if config else False
    
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Load HumanEval dataset."""
        logger.info("Loading HumanEval dataset...")
        
        try:
            dataset = hf_load_dataset("openai_humaneval")
            
            samples = []
            for item in dataset['test']:
                # Format input with or without test cases
                input_text = item['prompt']
                if self.include_tests:
                    input_text += f"\n\n# Test cases:\n{item['test']}"
                
                sample = DatasetSample(
                    instruction="Complete the following Python function:",
                    input=input_text,
                    output=item['canonical_solution'],
                    category='code_generation',
                    metadata={
                        'task_id': item['task_id'],
                        'test': item['test'],
                        'entry_point': item['entry_point']
                    }
                )
                samples.append(sample)
            
            # HumanEval is primarily for evaluation
            if self.eval_only:
                self.train_data = []
                self.eval_data = samples
            else:
                # Use 80/20 split if training is needed
                split_idx = int(len(samples) * 0.8)
                random.shuffle(samples)
                self.train_data = samples[:split_idx]
                self.eval_data = samples[split_idx:]
            
            logger.info(f"Loaded HumanEval: {len(self.train_data)} train, {len(self.eval_data)} eval")
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            raise
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="HumanEval",
            category="code_generation",
            num_train=len(self.train_data),
            num_eval=len(self.eval_data),
            description="164 hand-written programming problems for evaluating code generation",
            source="https://github.com/openai/human-eval",
            license="MIT"
        )
    
    def format_prompt(self, sample: DatasetSample, include_solution: bool = True) -> str:
        """
        Format prompt for code generation.
        
        Args:
            sample: Dataset sample
            include_solution: Whether to include the solution (for training)
        """
        if include_solution:
            return f"""### Instruction:
{sample.instruction}

### Code:
{sample.input}

### Solution:
{sample.output}"""
        else:
            return f"""### Instruction:
{sample.instruction}

### Code:
{sample.input}

### Solution:
"""


class MBPPDataset(BaseDatasetLoader):
    """
    MBPP (Mostly Basic Python Problems) dataset.
    
    Source: https://github.com/google-research/google-research/tree/master/mbpp
    Format: Problem description → Python solution
    Size: ~1000 problems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.include_tests = config.get('include_tests', True) if config else True
    
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Load MBPP dataset."""
        logger.info("Loading MBPP dataset...")
        
        try:
            dataset = hf_load_dataset("mbpp")
            
            train_samples = []
            for item in dataset['train']:
                input_text = item['text']
                if self.include_tests and item.get('test_list'):
                    test_examples = '\n'.join(item['test_list'][:3])  # Include first 3 tests
                    input_text += f"\n\nTest examples:\n{test_examples}"
                
                sample = DatasetSample(
                    instruction="Write a Python function to solve the following problem:",
                    input=input_text,
                    output=item['code'],
                    category='code_generation',
                    metadata={
                        'task_id': item['task_id'],
                        'test_list': item.get('test_list', []),
                        'test_setup_code': item.get('test_setup_code', ''),
                        'challenge_test_list': item.get('challenge_test_list', [])
                    }
                )
                train_samples.append(sample)
            
            eval_samples = []
            for item in dataset['test']:
                input_text = item['text']
                if self.include_tests and item.get('test_list'):
                    test_examples = '\n'.join(item['test_list'][:3])
                    input_text += f"\n\nTest examples:\n{test_examples}"
                
                sample = DatasetSample(
                    instruction="Write a Python function to solve the following problem:",
                    input=input_text,
                    output=item['code'],
                    category='code_generation',
                    metadata={
                        'task_id': item['task_id'],
                        'test_list': item.get('test_list', []),
                        'test_setup_code': item.get('test_setup_code', ''),
                        'challenge_test_list': item.get('challenge_test_list', [])
                    }
                )
                eval_samples.append(sample)
            
            self.train_data = train_samples
            self.eval_data = eval_samples
            
            logger.info(f"Loaded MBPP: {len(self.train_data)} train, {len(self.eval_data)} eval")
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load MBPP: {e}")
            raise
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="MBPP",
            category="code_generation",
            num_train=len(self.train_data),
            num_eval=len(self.eval_data),
            description="974 Mostly Basic Python Problems for code generation",
            source="https://github.com/google-research/google-research/tree/master/mbpp",
            license="CC-BY-4.0"
        )


class CodeAlpacaDataset(BaseDatasetLoader):
    """
    CodeAlpaca - Instruction-following dataset for code.
    
    Source: https://github.com/sahil280114/codealpaca
    Format: Instruction → Code response
    Size: 20K examples
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_samples = config.get('max_samples', None) if config else None
        self.train_split = config.get('train_split', 0.9) if config else 0.9
    
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Load CodeAlpaca dataset."""
        logger.info("Loading CodeAlpaca dataset...")
        
        try:
            dataset = hf_load_dataset("sahil2801/CodeAlpaca-20k")
            
            samples = []
            for item in dataset['train']:
                sample = DatasetSample(
                    instruction=item['instruction'],
                    input=item.get('input', ''),
                    output=item['output'],
                    category='code_generation',
                    metadata={}
                )
                samples.append(sample)
            
            # Limit samples if specified
            if self.max_samples and len(samples) > self.max_samples:
                random.shuffle(samples)
                samples = samples[:self.max_samples]
            
            # Split into train/eval
            split_idx = int(len(samples) * self.train_split)
            random.shuffle(samples)
            self.train_data = samples[:split_idx]
            self.eval_data = samples[split_idx:]
            
            logger.info(f"Loaded CodeAlpaca: {len(self.train_data)} train, {len(self.eval_data)} eval")
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load CodeAlpaca: {e}")
            raise
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="CodeAlpaca",
            category="code_generation",
            num_train=len(self.train_data),
            num_eval=len(self.eval_data),
            description="20K instruction-following examples for code generation",
            source="https://github.com/sahil280114/codealpaca",
            license="CC-BY-NC-4.0"
        )


class CodeContestsDataset(BaseDatasetLoader):
    """
    CodeContests - Competitive programming problems.
    
    Source: https://github.com/deepmind/code_contests
    Format: Problem description → Multiple solutions
    Difficulty: Advanced (competitive programming level)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.difficulty_filter = config.get('difficulty_filter', None) if config else None
        self.max_samples = config.get('max_samples', 5000) if config else 5000
        self.language = config.get('language', 'python') if config else 'python'
    
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Load CodeContests dataset."""
        logger.info(f"Loading CodeContests dataset (language: {self.language})...")
        
        try:
            dataset = hf_load_dataset("deepmind/code_contests")
            
            samples = []
            for split in ['train', 'valid', 'test']:
                for item in dataset[split]:
                    # Filter by difficulty if specified
                    if self.difficulty_filter:
                        difficulty = item.get('difficulty', '')
                        if difficulty.lower() != self.difficulty_filter.lower():
                            continue
                    
                    # Extract solutions for specified language
                    solutions = item.get('solutions', {})
                    if self.language not in solutions:
                        continue
                    
                    lang_solutions = solutions[self.language]
                    if not lang_solutions:
                        continue
                    
                    # Use first solution as canonical
                    canonical_solution = lang_solutions[0]
                    
                    # Format problem description
                    problem_desc = item.get('description', '')
                    input_desc = item.get('public_tests', {}).get('input', [''])[0]
                    output_desc = item.get('public_tests', {}).get('output', [''])[0]
                    
                    input_text = f"{problem_desc}\n\n"
                    if input_desc and output_desc:
                        input_text += f"Example:\nInput: {input_desc}\nOutput: {output_desc}"
                    
                    sample = DatasetSample(
                        instruction=f"Solve the following {self.language} programming problem:",
                        input=input_text,
                        output=canonical_solution,
                        category='code_generation_advanced',
                        metadata={
                            'name': item.get('name', ''),
                            'difficulty': item.get('difficulty', ''),
                            'source': item.get('source', ''),
                            'num_solutions': len(lang_solutions),
                            'time_limit': item.get('time_limit', {}),
                            'memory_limit': item.get('memory_limit', {})
                        }
                    )
                    samples.append(sample)
                    
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
                
                if self.max_samples and len(samples) >= self.max_samples:
                    break
            
            # Split 80/20
            split_idx = int(len(samples) * 0.8)
            random.shuffle(samples)
            self.train_data = samples[:split_idx]
            self.eval_data = samples[split_idx:]
            
            logger.info(f"Loaded CodeContests: {len(self.train_data)} train, {len(self.eval_data)} eval")
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load CodeContests: {e}")
            raise
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=f"CodeContests-{self.language}",
            category="code_generation",
            num_train=len(self.train_data),
            num_eval=len(self.eval_data),
            description=f"Competitive programming problems in {self.language}",
            source="https://github.com/deepmind/code_contests",
            license="Apache-2.0"
        )


# Factory function for easy dataset loading
def load_code_dataset(
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseDatasetLoader:
    """
    Factory function to load code generation datasets.
    
    Args:
        dataset_name: One of ['humaneval', 'mbpp', 'codealpaca', 'codecontests']
        config: Dataset-specific configuration
    
    Returns:
        Loaded dataset instance
    
    Example:
        >>> dataset = load_code_dataset('mbpp', {'include_tests': True})
        >>> train, eval = dataset.load()
        >>> print(dataset.get_info())
    """
    datasets_map = {
        'humaneval': HumanEvalDataset,
        'mbpp': MBPPDataset,
        'codealpaca': CodeAlpacaDataset,
        'codecontests': CodeContestsDataset
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in datasets_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets_map.keys())}")
    
    dataset_class = datasets_map[dataset_name]
    return dataset_class(config)