"""FLOPs estimation utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_flops(model) -> Optional[float]:
    """
    Estimate FLOPs per token for transformer models.
    
    Args:
        model: The language model
        
    Returns:
        FLOPs per token in GFLOPs, or None if estimation fails
    """
    try:
        cfg = model.config
        hidden_size = cfg.hidden_size
        intermediate_size = getattr(cfg, 'intermediate_size', hidden_size * 4)
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        head_dim = hidden_size // num_heads
        vocab_size = cfg.vocab_size
        
        qkv_proj = 3 * hidden_size * hidden_size * 2
        attn_matmul = num_heads * head_dim * 2
        output_proj = hidden_size * hidden_size * 2
        attention_flops = qkv_proj + attn_matmul + output_proj
        
        ffn_flops = (hidden_size * intermediate_size * 2 +
                    intermediate_size * hidden_size * 2)
        
        per_layer_flops = attention_flops + ffn_flops
        total_flops = per_layer_flops * num_layers + hidden_size * vocab_size * 2
        
        return total_flops / 1e9
    except AttributeError as e:
        logger.warning(f"Could not estimate FLOPs (non-transformer model?): {e}")
        return None


def calculate_mfu(
    flops_per_token: float,
    throughput: float,
    is_cuda: bool,
    peak_tflops: float
) -> Optional[float]:
    """
    Calculate Model FLOPs Utilization.
    
    Args:
        flops_per_token: FLOPs per token in GFLOPs
        throughput: Throughput in tokens/second
        is_cuda: Whether using CUDA
        peak_tflops: Peak TFLOPs of the device
        
    Returns:
        MFU percentage, or None if not on GPU
    """
    if not is_cuda or peak_tflops == 0:
        return None
    
    peak_gflops = peak_tflops * 1000
    achieved_gflops = flops_per_token * throughput
    mfu = (achieved_gflops / peak_gflops) * 100 if peak_gflops > 0 else 0
    
    return mfu