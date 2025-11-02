"""
Models module - Unified interface for all model types.

Supported models:
- HuggingFace (FP16, BF16, INT8, NF4 via bitsandbytes)
- GPTQ (4-bit via auto-gptq)
- AWQ (4-bit via autoawq)
- HQQ (2/3/4/8-bit via hqq)
"""

from models.model_interface import ModelInterface, create_model_interface
from models.huggingface_model import HuggingFaceModel
from models.gptq_model import GPTQModel
from models.awq_model import AWQModel
from models.hqq_model import HQQModel


__all__ = [
    'ModelInterface',
    'create_model_interface',
    'HuggingFaceModel',
    'GPTQModel',
    'AWQModel',
    'HQQModel',
]