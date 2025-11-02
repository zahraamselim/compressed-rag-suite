"""
Finetuning module for compressed models.

Supports:
- LoRA/QLoRA for memory-efficient finetuning
- Multiple quantization methods (NF4, INT8, GPTQ, AWQ, HQQ)
- Comparison of pre/post finetuning performance
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import json
import time

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class FinetuneResults:
    """Results from finetuning run."""
    # Training metrics
    train_loss: float
    eval_loss: Optional[float] = None
    train_runtime: float = 0.0
    train_samples_per_second: float = 0.0
    train_steps_per_second: float = 0.0
    
    # Model info
    model_name: str = ""
    quantization_method: str = ""
    bits_per_param: Optional[float] = None
    model_size_gb: float = 0.0
    
    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Performance comparison
    pre_finetune_metrics: Optional[Dict[str, float]] = None
    post_finetune_metrics: Optional[Dict[str, float]] = None
    improvement: Optional[Dict[str, float]] = None
    
    # Training config
    num_train_samples: int = 0
    num_eval_samples: int = 0
    num_epochs: int = 0
    learning_rate: float = 0.0
    batch_size: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, filepath: str):
        """Save results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {filepath}")


class QuantizedModelTrainer:
    """
    Trainer for finetuning quantized models with LoRA/QLoRA.
    
    Supports comparison of different quantization methods:
    - NF4 (4-bit NormalFloat)
    - INT8 (8-bit integer)
    - GPTQ (Gradient-based Post-Training Quantization)
    - AWQ (Activation-aware Weight Quantization)
    - HQQ (Half-Quadratic Quantization)
    """
    
    def __init__(
        self,
        model_interface,
        tokenizer,
        output_dir: str = "./finetuned_models",
        use_lora: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model_interface: ModelInterface instance with loaded model
            tokenizer: Tokenizer instance
            output_dir: Directory to save finetuned models
            use_lora: Whether to use LoRA (recommended for quantized models)
        """
        self.model_interface = model_interface
        self.model = model_interface.get_model()
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.use_lora = use_lora
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect quantization method
        self.quantization_method = self._detect_quantization()
        logger.info(f"Detected quantization: {self.quantization_method}")
    
    def _detect_quantization(self) -> str:
        """Detect which quantization method is being used."""
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # Check for bitsandbytes quantization
            if hasattr(config, 'quantization_config'):
                quant_config = config.quantization_config
                if hasattr(quant_config, 'quant_method'):
                    return quant_config.quant_method
                if hasattr(quant_config, 'load_in_4bit') and quant_config.load_in_4bit:
                    return 'nf4'
                if hasattr(quant_config, 'load_in_8bit') and quant_config.load_in_8bit:
                    return 'int8'
            
            # Check for GPTQ
            if hasattr(config, 'quantization_config') and 'gptq' in str(config.quantization_config).lower():
                return 'gptq'
        
        return 'none'
    
    def setup_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        task_type: str = "CAUSAL_LM"
    ) -> None:
        """
        Setup LoRA configuration for the model.
        
        Args:
            r: LoRA attention dimension
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Which modules to apply LoRA to (None = auto-detect)
            task_type: Task type for LoRA
        """
        logger.info("Setting up LoRA configuration...")
        
        # Auto-detect target modules if not specified
        if target_modules is None:
            # Common target modules for different architectures
            if 'llama' in self.model.config.model_type.lower() or 'mistral' in self.model.config.model_type.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]  # Conservative default
        
        logger.info(f"LoRA target modules: {target_modules}")
        
        # Prepare model for k-bit training if quantized
        if self.quantization_method in ['nf4', 'int8', 'gptq', 'awq']:
            logger.info("Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")
        
        self.lora_config = lora_config
    
    def prepare_dataset(
        self,
        train_samples: List,
        eval_samples: Optional[List] = None,
        max_length: int = 512
    ) -> tuple:
        """
        Prepare datasets for training.
        
        Args:
            train_samples: List of DatasetSample objects
            eval_samples: List of DatasetSample objects for evaluation
            max_length: Maximum sequence length
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info("Preparing datasets...")
        
        def format_sample(sample):
            """Format a single sample."""
            if sample.input:
                text = f"### Instruction:\n{sample.instruction}\n\n### Input:\n{sample.input}\n\n### Response:\n{sample.output}"
            else:
                text = f"### Instruction:\n{sample.instruction}\n\n### Response:\n{sample.output}"
            return text
        
        def tokenize_function(examples):
            """Tokenize samples."""
            texts = [format_sample(sample) for sample in examples['samples']]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict({'samples': train_samples})
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['samples']
        )
        
        eval_dataset = None
        if eval_samples:
            eval_dataset = Dataset.from_dict({'samples': eval_samples})
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['samples']
            )
        
        logger.info(f"Prepared {len(train_dataset)} training samples")
        if eval_dataset:
            logger.info(f"Prepared {len(eval_dataset)} evaluation samples")
        
        return train_dataset, eval_dataset
    
    def finetune(
        self,
        train_dataset,
        eval_dataset: Optional = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: Optional[int] = None,
        fp16: bool = True,
        bf16: bool = False,
        max_grad_norm: float = 1.0,
        output_name: Optional[str] = None
    ) -> FinetuneResults:
        """
        Finetune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps (None = eval at end of epoch)
            fp16: Use FP16 training
            bf16: Use BF16 training (recommended for Ampere+ GPUs)
            max_grad_norm: Max gradient norm for clipping
            output_name: Name for output directory
        
        Returns:
            FinetuneResults object
        """
        logger.info("="*80)
        logger.info("STARTING FINETUNING")
        logger.info("="*80)
        
        if output_name is None:
            output_name = f"{self.quantization_method}_{int(time.time())}"
        
        output_path = self.output_dir / output_name
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_steps else save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            fp16=fp16 and torch.cuda.is_available(),
            bf16=bf16 and torch.cuda.is_available(),
            max_grad_norm=max_grad_norm,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_path}")
        trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)
        
        # Compile results
        results = FinetuneResults(
            train_loss=train_result.training_loss,
            train_runtime=train_result.metrics['train_runtime'],
            train_samples_per_second=train_result.metrics['train_samples_per_second'],
            train_steps_per_second=train_result.metrics['train_steps_per_second'],
            model_name=self.model_interface.model_path,
            quantization_method=self.quantization_method,
            num_train_samples=len(train_dataset),
            num_eval_samples=len(eval_dataset) if eval_dataset else 0,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        
        # Add LoRA config if used
        if self.use_lora and hasattr(self, 'lora_config'):
            results.lora_r = self.lora_config.r
            results.lora_alpha = self.lora_config.lora_alpha
            results.lora_dropout = self.lora_config.lora_dropout
            results.target_modules = self.lora_config.target_modules
        
        # Evaluate if eval dataset provided
        if eval_dataset:
            logger.info("Running evaluation...")
            eval_result = trainer.evaluate()
            results.eval_loss = eval_result['eval_loss']
        
        # Save results
        results.save(output_path / "finetune_results.json")
        
        logger.info("="*80)
        logger.info("FINETUNING COMPLETE")
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"Train loss: {results.train_loss:.4f}")
        if results.eval_loss:
            logger.info(f"Eval loss: {results.eval_loss:.4f}")
        logger.info("="*80)
        
        return results
    
    def compare_pre_post_finetune(
        self,
        eval_function,
        pre_model_path: Optional[str] = None,
        results: Optional[FinetuneResults] = None
    ) -> FinetuneResults:
        """
        Compare model performance before and after finetuning.
        
        Args:
            eval_function: Function that takes model and returns metrics dict
            pre_model_path: Path to pre-finetuned model (if different)
            results: FinetuneResults object to update
        
        Returns:
            Updated FinetuneResults with comparison metrics
        """
        logger.info("Comparing pre/post finetuning performance...")
        
        # Evaluate current (post-finetune) model
        logger.info("Evaluating finetuned model...")
        post_metrics = eval_function(self.model)
        
        # Load and evaluate pre-finetune model if path provided
        pre_metrics = None
        if pre_model_path:
            logger.info(f"Loading pre-finetuned model from {pre_model_path}")
            # TODO: Implement model loading and evaluation
            pass
        
        # Calculate improvement
        improvement = {}
        if pre_metrics and post_metrics:
            for key in post_metrics:
                if key in pre_metrics:
                    pre_val = pre_metrics[key]
                    post_val = post_metrics[key]
                    improvement[key] = post_val - pre_val
        
        # Update results
        if results:
            results.pre_finetune_metrics = pre_metrics
            results.post_finetune_metrics = post_metrics
            results.improvement = improvement
        
        logger.info("Performance comparison:")
        for key, value in post_metrics.items():
            pre_val = pre_metrics.get(key, 0) if pre_metrics else 0
            imp = improvement.get(key, 0)
            logger.info(f"  {key}: {pre_val:.4f} → {post_val:.4f} ({imp:+.4f})")
        
        return results