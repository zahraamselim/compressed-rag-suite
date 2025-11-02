"""
Training module for quantized models with integrated evaluation.

Supports:
- LoRA/QLoRA finetuning for quantized models
- Parameter-efficient fine-tuning (PEFT)
- Post-epoch evaluation using evaluation system
- Best model tracking based on evaluation metrics
- Checkpoint management for long training sessions
"""

import torch
from typing import Optional, Dict, Any, List, Callable
import logging
from pathlib import Path
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from datasets import Dataset
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationCallback(TrainerCallback):
    """Callback to run evaluation benchmarks after each epoch."""
    
    def __init__(
        self,
        model_interface,
        eval_config: Dict[str, Any],
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./eval_results",
        run_efficiency: bool = False,
        run_performance: bool = True,
        run_retrieval: bool = False,
        rag_pipeline = None,
        save_detailed: bool = True
    ):
        """
        Initialize evaluation callback.
        
        Args:
            model_interface: ModelInterface instance
            eval_config: Evaluation configuration
            eval_dataset: Optional evaluation dataset for perplexity
            output_dir: Directory to save evaluation results
            run_efficiency: Run efficiency benchmarks
            run_performance: Run performance benchmarks
            run_retrieval: Run retrieval benchmarks
            rag_pipeline: RAGPipeline for retrieval eval
            save_detailed: Save detailed evaluation results
        """
        self.model_interface = model_interface
        self.eval_config = eval_config
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_efficiency = run_efficiency
        self.run_performance = run_performance
        self.run_retrieval = run_retrieval
        self.rag_pipeline = rag_pipeline
        self.save_detailed = save_detailed
        
        # Track evaluation history
        self.eval_history = []
        self.best_metric_value = None
        self.best_epoch = None
        
        logger.info(f"Evaluation callback initialized (output: {output_dir})")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation at the end of each epoch."""
        epoch = int(state.epoch)
        logger.info(f"\n{'='*80}")
        logger.info(f"Running post-epoch evaluation (Epoch {epoch})")
        logger.info(f"{'='*80}")
        
        try:
            # Import here to avoid circular imports
            from evaluation.runner import EvaluationRunner
            
            # Create evaluation runner
            runner = EvaluationRunner(
                model_interface=self.model_interface,
                config={'evaluation': self.eval_config},
                rag_pipeline=self.rag_pipeline,
                verbose=False
            )
            
            # Prepare output directory for this epoch
            epoch_dir = self.output_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            
            # Run selected benchmarks
            results = runner.run_all(
                run_efficiency=self.run_efficiency,
                run_performance=self.run_performance,
                run_retrieval=self.run_retrieval,
                save_results=self.save_detailed,
                output_dir=str(epoch_dir)
            )
            
            # Extract key metrics
            summary = results.get_summary()
            summary['epoch'] = epoch
            summary['global_step'] = state.global_step
            
            # Log key metrics
            logger.info("\nKey Metrics:")
            for key, value in summary.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    logger.info(f"  {key}: {value}")
            
            # Store in history
            self.eval_history.append(summary)
            
            # Save evaluation history
            history_path = self.output_dir / "evaluation_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.eval_history, f, indent=2)
            
            # Track best model
            self._update_best_model(summary, epoch)
            
            logger.info(f"Evaluation complete for epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Evaluation failed at epoch {epoch}: {e}", exc_info=True)
    
    def _update_best_model(self, summary: Dict[str, Any], epoch: int):
        """Track the best model based on evaluation metrics."""
        # Define metric to optimize (you can make this configurable)
        metric_name = 'average_accuracy'  # or 'f1_score', 'exact_match', etc.
        higher_is_better = True
        
        if metric_name not in summary:
            # Try alternative metrics
            for alt_metric in ['f1_score', 'exact_match', 'perplexity']:
                if alt_metric in summary:
                    metric_name = alt_metric
                    higher_is_better = alt_metric != 'perplexity'
                    break
        
        if metric_name in summary:
            current_value = summary[metric_name]
            
            if self.best_metric_value is None:
                self.best_metric_value = current_value
                self.best_epoch = epoch
                logger.info(f"✓ New best {metric_name}: {current_value:.4f} (epoch {epoch})")
            else:
                is_better = (
                    (higher_is_better and current_value > self.best_metric_value) or
                    (not higher_is_better and current_value < self.best_metric_value)
                )
                
                if is_better:
                    self.best_metric_value = current_value
                    self.best_epoch = epoch
                    logger.info(f"✓ New best {metric_name}: {current_value:.4f} (epoch {epoch})")
                else:
                    logger.info(f"  Best {metric_name} remains: {self.best_metric_value:.4f} (epoch {self.best_epoch})")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate final evaluation report."""
        logger.info(f"\n{'='*80}")
        logger.info("Training complete! Generating evaluation report...")
        logger.info(f"{'='*80}")
        
        if self.eval_history:
            # Save final summary
            final_summary = {
                'total_epochs': len(self.eval_history),
                'best_epoch': self.best_epoch,
                'best_metric_value': self.best_metric_value,
                'evaluation_history': self.eval_history
            }
            
            summary_path = self.output_dir / "final_evaluation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2)
            
            logger.info(f"Evaluation history saved to {self.output_dir}")
            if self.best_epoch is not None:
                logger.info(f"Best model at epoch {self.best_epoch} with metric {self.best_metric_value:.4f}")


class QuantizedModelFinetuner:
    """
    Finetune quantized models using QLoRA with integrated evaluation.
    
    Features:
    - Automatic LoRA configuration for 4-bit models
    - Post-epoch evaluation using evaluation system
    - Best model tracking based on evaluation metrics
    - Checkpoint recovery for long training sessions
    - Memory-efficient training
    - Support for instruction-tuning format
    """
    
    def __init__(
        self,
        model_interface,
        output_dir: str,
        lora_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize finetuner.
        
        Args:
            model_interface: ModelInterface instance with loaded model
            output_dir: Directory to save checkpoints and final model
            lora_config: LoRA configuration dict
            training_config: Training arguments dict
            evaluation_config: Evaluation configuration dict
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
        
        # Default evaluation config
        self.evaluation_config = evaluation_config or {
            'enabled': False,
            'run_every_epoch': True,
            'run_efficiency': False,
            'run_performance': True,
            'run_retrieval': False,
            'efficiency': {},
            'performance': {
                'perplexity': {'enabled': True, 'num_samples': 50},
                'lm_eval': {'enabled': False}
            },
            'retrieval': {}
        }
        
        self.peft_model = None
        self.tokenizer = None
        self.trainer = None
        self.evaluation_callback = None
    
    def prepare_model(self):
        """Prepare model for QLoRA training using ModelInterface."""
        logger.info("Preparing model for QLoRA finetuning...")
        
        # Get base model and tokenizer from interface
        base_model = self.model_interface.get_model()
        self.tokenizer = self.model_interface.get_tokenizer()
        
        # Check if model is already a PEFT model
        if isinstance(base_model, PeftModel):
            logger.info("Model is already a PEFT model, using as-is")
            self.peft_model = base_model
        else:
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
            self.peft_model = get_peft_model(base_model, peft_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        logger.info(f"Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
        logger.info(f"Total params: {total_params:,}")
        
        return self.peft_model
    
    def prepare_dataset(
        self,
        dataset_samples: List[Dict[str, str]],
        max_length: int = 512,
        format_fn: Optional[Callable] = None
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
    
    def setup_evaluation(
        self,
        eval_dataset: Optional[Dataset] = None,
        rag_pipeline = None
    ):
        """
        Setup evaluation callback for post-epoch evaluation.
        
        Args:
            eval_dataset: Optional evaluation dataset for perplexity
            rag_pipeline: Optional RAG pipeline for retrieval evaluation
        """
        if not self.evaluation_config.get('enabled', False):
            logger.info("Evaluation disabled in config")
            return
        
        if not self.evaluation_config.get('run_every_epoch', True):
            logger.info("Post-epoch evaluation disabled")
            return
        
        logger.info("Setting up evaluation callback...")
        
        eval_output_dir = self.output_dir / "evaluations"
        
        self.evaluation_callback = EvaluationCallback(
            model_interface=self.model_interface,
            eval_config=self.evaluation_config,
            eval_dataset=eval_dataset,
            output_dir=str(eval_output_dir),
            run_efficiency=self.evaluation_config.get('run_efficiency', False),
            run_performance=self.evaluation_config.get('run_performance', True),
            run_retrieval=self.evaluation_config.get('run_retrieval', False),
            rag_pipeline=rag_pipeline,
            save_detailed=True
        )
        
        logger.info(f"Evaluation will run after each epoch")
        logger.info(f"Results will be saved to {eval_output_dir}")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: bool = True,
        rag_pipeline = None
    ):
        """
        Train the model with optional post-epoch evaluation.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Whether to resume from last checkpoint
            rag_pipeline: Optional RAG pipeline for retrieval evaluation
        """
        if self.peft_model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        logger.info("Setting up training...")
        
        # Setup evaluation if enabled
        self.setup_evaluation(eval_dataset, rag_pipeline)
        
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
        
        # Prepare callbacks
        callbacks = []
        if self.evaluation_callback is not None:
            callbacks.append(self.evaluation_callback)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None
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
        
        # Print evaluation summary if available
        if self.evaluation_callback and self.evaluation_callback.eval_history:
            logger.info("\n" + "="*80)
            logger.info("EVALUATION SUMMARY")
            logger.info("="*80)
            logger.info(f"Total epochs evaluated: {len(self.evaluation_callback.eval_history)}")
            if self.evaluation_callback.best_epoch is not None:
                logger.info(f"Best model: Epoch {self.evaluation_callback.best_epoch}")
                logger.info(f"Best metric: {self.evaluation_callback.best_metric_value:.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save the finetuned model."""
        save_path = Path(path) if path else self.output_dir / 'final_model'
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}...")
        
        # Save LoRA adapters
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configs
        with open(save_path / 'training_config.json', 'w') as f:
            json.dump({
                'lora_config': self.lora_config,
                'training_config': self.training_config,
                'evaluation_config': self.evaluation_config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
        return str(save_path)
    
    def load_finetuned_model(self, path: str):
        """Load a finetuned model into the model interface."""
        logger.info(f"Loading finetuned model from {path}...")
        
        base_model = self.model_interface.get_model()
        self.peft_model = PeftModel.from_pretrained(base_model, path)
        self.tokenizer = self.model_interface.get_tokenizer()
        
        # Update the model in the interface
        # Note: This is a bit hacky, but necessary to keep interface in sync
        self.model_interface.model = self.peft_model
        
        logger.info("Finetuned model loaded!")
        return self.peft_model


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