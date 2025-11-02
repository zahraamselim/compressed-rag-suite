"""
Domain Expertise Datasets

Available datasets:
1. MedQA - Medical question answering
2. LegalBench - Legal reasoning tasks
3. ArXiv Papers - Scientific domain expertise (custom)
"""

from pathlib import Path
import json


class MedQADataset(BaseDatasetLoader):
    """
    MedQA - Medical Question Answering dataset.
    
    Source: https://github.com/jind11/MedQA
    Format: Medical exam questions with multiple choice answers
    """
    
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Load MedQA dataset."""
        logger.info("Loading MedQA dataset...")
        
        try:
            # MedQA US (English)
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")
            
            train_samples = []
            for item in dataset['train']:
                # Format multiple choice
                choices_text = "\n".join([
                    f"{chr(65+i)}. {choice}"
                    for i, choice in enumerate(item['choices'])
                ])
                
                sample = DatasetSample(
                    instruction="Answer the following medical question by selecting the correct option:",
                    input=f"{item['question']}\n\n{choices_text}",
                    output=item['answer'][0] if item['answer'] else "",
                    category='domain_expertise_medical',
                    metadata={
                        'id': item['id'],
                        'type': item['type']
                    }
                )
                train_samples.append(sample)
            
            eval_samples = []
            for item in dataset['test']:
                choices_text = "\n".join([
                    f"{chr(65+i)}. {choice}"
                    for i, choice in enumerate(item['choices'])
                ])
                
                sample = DatasetSample(
                    instruction="Answer the following medical question by selecting the correct option:",
                    input=f"{item['question']}\n\n{choices_text}",
                    output=item['answer'][0] if item['answer'] else "",
                    category='domain_expertise_medical',
                    metadata={
                        'id': item['id'],
                        'type': item['type']
                    }
                )
                eval_samples.append(sample)
            
            self.train_data = train_samples
            self.eval_data = eval_samples
            
            logger.info(f"Loaded MedQA: {len(self.train_data)} train, {len(self.eval_data)} eval")
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load MedQA: {e}")
            raise
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="MedQA",
            category="domain_expertise",
            num_train=len(self.train_data),
            num_eval=len(self.eval_data),
            description="Medical question answering from USMLE exams",
            source="https://github.com/jind11/MedQA",
            license="Unknown"
        )


class LegalBenchDataset(BaseDatasetLoader):
    """
    LegalBench - Legal reasoning tasks.
    
    Source: https://github.com/HazyResearch/legalbench
    Format: Various legal reasoning tasks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_name = config.get('task_name', 'abercrombie') if config else 'abercrombie'
    
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Load LegalBench dataset."""
        logger.info(f"Loading LegalBench task: {self.task_name}...")
        
        try:
            dataset = load_dataset("nguha/legalbench", self.task_name)
            
            samples = []
            for item in dataset['test']:
                sample = DatasetSample(
                    instruction=item['instruction'] if 'instruction' in item else "Analyze the following legal text:",
                    input=item['text'],
                    output=str(item['label']),
                    category='domain_expertise_legal',
                    metadata={
                        'task': self.task_name
                    }
                )
                samples.append(sample)
            
            # Split 80/20 for train/eval
            split_idx = int(len(samples) * 0.8)
            random.shuffle(samples)
            self.train_data = samples[:split_idx]
            self.eval_data = samples[split_idx:]
            
            logger.info(f"Loaded LegalBench: {len(self.train_data)} train, {len(self.eval_data)} eval")
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load LegalBench: {e}")
            raise
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=f"LegalBench-{self.task_name}",
            category="domain_expertise",
            num_train=len(self.train_data),
            num_eval=len(self.eval_data),
            description=f"Legal reasoning task: {self.task_name}",
            source="https://github.com/HazyResearch/legalbench",
            license="CC-BY-4.0"
        )


class ArXivDomainDataset(BaseDatasetLoader):
    """
    Custom ArXiv Papers Dataset for Domain Expertise.
    
    Uses ArXiv papers to create domain-specific QA pairs.
    You'll need to provide processed papers or use the builder utility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.data_path = config.get('data_path') if config else None
        self.domain = config.get('domain', 'cs') if config else 'cs'  # cs, physics, math, etc.
    
    def load(self) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Load custom ArXiv dataset."""
        if not self.data_path:
            raise ValueError("data_path required for ArXivDomainDataset")
        
        logger.info(f"Loading ArXiv domain dataset from {self.data_path}...")
        
        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = DatasetSample(
                instruction=item['instruction'],
                input=item.get('input', ''),
                output=item['output'],
                category=f'domain_expertise_{self.domain}',
                metadata=item.get('metadata', {})
            )
            samples.append(sample)
        
        # Split 80/20
        split_idx = int(len(samples) * 0.8)
        random.shuffle(samples)
        self.train_data = samples[:split_idx]
        self.eval_data = samples[split_idx:]
        
        logger.info(f"Loaded ArXiv dataset: {len(self.train_data)} train, {len(self.eval_data)} eval")
        return self.train_data, self.eval_data
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=f"ArXiv-{self.domain}",
            category="domain_expertise",
            num_train=len(self.train_data),
            num_eval=len(self.eval_data),
            description=f"Custom ArXiv dataset for {self.domain} domain",
            source="ArXiv papers",
            license="Varies by paper"
        )
