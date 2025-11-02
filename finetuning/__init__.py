"""
Finetuning module for compressed models.

Supports:
- LoRA/QLoRA for memory-efficient finetuning
- Multiple quantization methods (NF4, INT8, GPTQ, AWQ, HQQ)
- Pre/post finetuning performance comparison
"""

from finetuning.trainer import QuantizedModelTrainer, FinetuneResults

__all__ = [
    'QuantizedModelTrainer',
    'FinetuneResults',
]
