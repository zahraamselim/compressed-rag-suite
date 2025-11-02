"""
Training module for quantized models.

Supports:
- LoRA/QLoRA finetuning for quantized models
- Parameter-efficient fine-tuning (PEFT)
- Training with custom datasets
- Checkpoint management for long training sessions
"""

import torch
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class QuantizedModelFinetuner:
    """
    Finetune quantized models using QLoRA.
    
    Features:
    - Automatic LoRA configuration for 4-bit models
    - Checkpoint recovery for long training sessions
    - Memory-efficient training
    - Support for instruction-tuning format
    """
    
    def __init__(
        self,
        model_interface,
        output_dir: str,
        lora_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize finetuner.
        
        Args:
            model_interface: ModelInterface instance with loaded model
            output_dir: Directory to save checkpoints and final model
            lora_config: LoRA configuration dict
            training_config: Training arguments dict
        """
        self.model_interface = model_interface
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default LoRA config for code generation
        self.lora_config = lora_config or {
            'r': 16,
            'lora_alpha': 32,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            'lora_dropout': 0.05,
            'bias': 'none',
            'task_type': 'CAUSAL_LM'
        }
        
        # Default training config optimized for Kaggle (2 T4 GPUs, 9h limit)
        self.training_config = training_config or {
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2e-4,
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.03,
            'weight_decay': 0.001,
            'logging_steps': 10,
            'save_steps': 100,
            'save_total_limit': 3,
            'fp16': True,
            'optim': 'paged_adamw_8bit',
            'gradient_checkpointing': True,
            'max_grad_norm': 0.3,
        }
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def prepare_model(self):
        """Prepare model for QLoRA training."""
        logger.info("Preparing model for QLoRA finetuning...")
        
        # Get base model and tokenizer
        base_model = self.model_interface.get_model()
        self.tokenizer = self.model_interface.get_tokenizer()
        
        # Prepare model for k-bit training
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Enable gradient checkpointing
        if self.training_config.get('gradient_checkpointing', True):
            base_model.gradient_checkpointing_enable()
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            lora_dropout=self.lora_config['lora_dropout'],
            bias=self.lora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, peft_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        logger.info(f"Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
        logger.info(f"Total params: {total_params:,}")
        
        return self.model
    
    def prepare_dataset(
        self,
        dataset_samples: List[Dict[str, str]],
        max_length: int = 512,
        format_fn: Optional[callable] = None
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            dataset_samples: List of dicts with 'instruction', 'input', 'output'
            max_length: Maximum sequence length
            format_fn: Optional function to format samples
            
        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Preparing {len(dataset_samples)} samples...")
        
        # Default format function for instruction-following
        if format_fn is None:
            def format_fn(sample):
                if sample.get('input'):
                    prompt = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
                else:
                    prompt = f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""
                return prompt
        
        # Format all samples
        formatted_texts = [format_fn(s) for s in dataset_samples]
        
        # Tokenize
        def tokenize_fn(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )
        
        # Create dataset
        dataset = Dataset.from_dict({'text': formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing"
        )
        
        logger.info(f"Dataset prepared: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: bool = True
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Whether to resume from last checkpoint
        """
        if self.model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        logger.info("Setting up training...")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            lr_scheduler_type=self.training_config['lr_scheduler_type'],
            warmup_ratio=self.training_config['warmup_ratio'],
            weight_decay=self.training_config['weight_decay'],
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=self.training_config['logging_steps'],
            save_steps=self.training_config['save_steps'],
            save_total_limit=self.training_config['save_total_limit'],
            eval_strategy='steps' if eval_dataset else 'no',
            eval_steps=self.training_config.get('eval_steps', 100) if eval_dataset else None,
            fp16=self.training_config['fp16'],
            optim=self.training_config['optim'],
            max_grad_norm=self.training_config['max_grad_norm'],
            report_to='none',
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model='eval_loss' if eval_dataset else None,
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Find checkpoint if resuming
        checkpoint = None
        if resume_from_checkpoint:
            checkpoints = list(self.output_dir.glob('checkpoint-*'))
            if checkpoints:
                checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split('-')[1])))
                logger.info(f"Resuming from checkpoint: {checkpoint}")
        
        # Train
        logger.info("Starting training...")
        logger.info(f"Effective batch size: {self.training_config['per_device_train_batch_size'] * self.training_config['gradient_accumulation_steps']}")
        logger.info(f"Total steps: ~{len(train_dataset) // (self.training_config['per_device_train_batch_size'] * self.training_config['gradient_accumulation_steps']) * self.training_config['num_train_epochs']}")
        
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save metrics
        metrics = train_result.metrics
        self.trainer.save_metrics("train", metrics)
        
        logger.info("Training complete!")
        logger.info(f"Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save the finetuned model."""
        save_path = Path(path) if path else self.output_dir / 'final_model'
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}...")
        
        # Save LoRA adapters
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configs
        with open(save_path / 'training_config.json', 'w') as f:
            json.dump({
                'lora_config': self.lora_config,
                'training_config': self.training_config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
        return str(save_path)
    
    def load_finetuned_model(self, path: str):
        """Load a finetuned model."""
        from peft import PeftModel
        
        logger.info(f"Loading finetuned model from {path}...")
        
        base_model = self.model_interface.get_model()
        self.model = PeftModel.from_pretrained(base_model, path)
        self.tokenizer = self.model_interface.get_tokenizer()
        
        logger.info("Finetuned model loaded!")
        return self.model


def create_code_format_fn():
    """Create format function for code generation tasks."""
    def format_fn(sample):
        instruction = sample.get('instruction', 'Complete the following code:')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        if input_text:
            return f"""### Instruction:
{instruction}

### Code:
{input_text}

### Solution:
{output}"""
        else:
            return f"""### Instruction:
{instruction}

### Solution:
{output}"""
    
    return format_fn


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    seconds_per_step: float = 1.5
) -> Dict[str, float]:
    """
    Estimate training time.
    
    Args:
        num_samples: Number of training samples
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of epochs
        seconds_per_step: Estimated seconds per training step
        
    Returns:
        Dict with time estimates
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    estimated_seconds = total_steps * seconds_per_step
    estimated_hours = estimated_seconds / 3600
    
    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'estimated_seconds': estimated_seconds,
        'estimated_hours': estimated_hours,
        'effective_batch_size': effective_batch_size
    }