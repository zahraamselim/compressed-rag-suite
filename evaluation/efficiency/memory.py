"""Memory measurement utilities."""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def get_model_size(model) -> Tuple[float, int]:
    """
    Calculate model size.
    
    Args:
        model: The language model
        
    Returns:
        Tuple of (size_gb, size_bytes)
    """
    param_size = sum(p.element_size() * p.numel() for p in model.parameters())
    buffer_size = sum(b.element_size() * b.numel() for b in model.buffers())
    total_bytes = param_size + buffer_size
    size_gb = total_bytes / (1024**3)
    
    logger.info(f"Model size: {size_gb:.3f} GB")
    return size_gb, total_bytes


def get_bits_per_param(model) -> float:
    """
    Calculate bits per parameter.
    
    Args:
        model: The language model
        
    Returns:
        Average bits per parameter
    """
    sample_param = next(model.parameters())
    
    if hasattr(sample_param, 'quant_state'):
        quant_type = str(getattr(sample_param.quant_state, 'quant_type', 'unknown')).lower()
        
        if 'nf4' in quant_type or 'int4' in quant_type:
            theoretical_bits = 4.0
        elif 'int8' in quant_type or 'fp8' in quant_type:
            theoretical_bits = 8.0
        else:
            theoretical_bits = 16.0
        
        total_params = sum(p.numel() for p in model.parameters())
        _, size_bytes = get_model_size(model)
        actual_bits = (size_bytes * 8) / total_params if total_params > 0 else theoretical_bits
        
        return actual_bits
    else:
        dtype = sample_param.dtype
        dtype_bits = {
            torch.float32: 32.0,
            torch.float16: 16.0,
            torch.bfloat16: 16.0,
            torch.float64: 64.0,
            torch.int8: 8.0
        }
        return dtype_bits.get(dtype, 16.0)


def get_peak_memory(is_cuda: bool) -> float:
    """
    Get peak memory usage in MB.
    
    Args:
        is_cuda: Whether using CUDA
        
    Returns:
        Peak memory in MB
    """
    if is_cuda:
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0