"""FLOPs estimation utilities."""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def estimate_flops(model) -> Optional[float]:
    """
    Estimate FLOPs per token for transformer models.
    
    Based on the formula:
    FLOPs ≈ 2 * N * (attention + FFN) per token
    
    where N is number of layers and calculations include:
    - QKV projections
    - Attention computation
    - Output projection
    - Feed-forward network
    
    Args:
        model: The language model
        
    Returns:
        FLOPs per token in GFLOPs, or None if estimation fails
    """
    try:
        cfg = model.config
        
        # Get model dimensions
        hidden_size = cfg.hidden_size
        intermediate_size = getattr(cfg, 'intermediate_size', hidden_size * 4)
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        head_dim = hidden_size // num_heads
        vocab_size = cfg.vocab_size
        
        # Attention FLOPs per layer
        # QKV projection: 3 * (hidden_size * hidden_size * 2) multiply-accumulate ops
        qkv_proj = 3 * hidden_size * hidden_size * 2
        
        # Attention scores: num_heads * (seq_len * head_dim * 2)
        # Using seq_len=1 for per-token estimate
        attn_scores = num_heads * head_dim * 2
        
        # Attention output: num_heads * (seq_len * head_dim * 2)
        attn_output = num_heads * head_dim * 2
        
        # Output projection: hidden_size * hidden_size * 2
        output_proj = hidden_size * hidden_size * 2
        
        attention_flops = qkv_proj + attn_scores + attn_output + output_proj
        
        # Feed-forward network FLOPs per layer
        # Up projection: hidden_size * intermediate_size * 2
        # Down projection: intermediate_size * hidden_size * 2
        ffn_flops = (hidden_size * intermediate_size * 2 +
                    intermediate_size * hidden_size * 2)
        
        # Total per layer
        per_layer_flops = attention_flops + ffn_flops
        
        # Total for all layers plus final layer norm and LM head
        total_flops = per_layer_flops * num_layers + hidden_size * vocab_size * 2
        
        # Convert to GFLOPs
        gflops = total_flops / 1e9
        
        logger.info(f"Estimated FLOPs: {gflops:.2f} GFLOPs/token")
        logger.debug(f"  Attention FLOPs per layer: {attention_flops / 1e9:.2f} GFLOPs")
        logger.debug(f"  FFN FLOPs per layer: {ffn_flops / 1e9:.2f} GFLOPs")
        logger.debug(f"  Total layers: {num_layers}")
        
        return gflops
        
    except AttributeError as e:
        logger.warning(f"Could not estimate FLOPs (non-transformer architecture?): {e}")
        return None
    except Exception as e:
        logger.error(f"FLOPs estimation failed: {e}")
        return None


def calculate_mfu(
    flops_per_token: float,
    throughput: float,
    is_cuda: bool,
    peak_tflops: float
) -> Optional[float]:
    """
    Calculate Model FLOPs Utilization (MFU).
    
    MFU = (Achieved FLOPs) / (Peak Hardware FLOPs) * 100%
    
    where:
    - Achieved FLOPs = FLOPs per token * throughput (tokens/sec)
    - Peak Hardware FLOPs = theoretical maximum of the device
    
    Args:
        flops_per_token: FLOPs per token in GFLOPs
        throughput: Throughput in tokens/second
        is_cuda: Whether using CUDA
        peak_tflops: Peak TFLOPs of the device
        
    Returns:
        MFU percentage (0-100), or None if not on GPU
    """
    if not is_cuda or peak_tflops == 0:
        logger.debug("MFU calculation requires GPU with known peak TFLOPs")
        return None
    
    if flops_per_token <= 0 or throughput <= 0:
        logger.warning(f"Invalid inputs for MFU: flops={flops_per_token}, throughput={throughput}")
        return None
    
    # Convert peak to GFLOPs
    peak_gflops = peak_tflops * 1000
    
    # Calculate achieved GFLOPs
    achieved_gflops = flops_per_token * throughput
    
    # Calculate MFU percentage
    mfu = (achieved_gflops / peak_gflops) * 100 if peak_gflops > 0 else 0.0
    
    logger.info(f"Model FLOPs Utilization (MFU): {mfu:.2f}%")
    logger.debug(f"  Achieved: {achieved_gflops:.2f} GFLOPs/s")
    logger.debug(f"  Peak: {peak_gflops:.2f} GFLOPs/s")
    
    return mfu


def get_flops_breakdown(model) -> Optional[Dict[str, float]]:
    """
    Get detailed FLOPs breakdown by component.
    
    Args:
        model: The language model
        
    Returns:
        Dictionary with FLOPs breakdown in GFLOPs, or None if fails
    """
    try:
        cfg = model.config
        
        hidden_size = cfg.hidden_size
        intermediate_size = getattr(cfg, 'intermediate_size', hidden_size * 4)
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        head_dim = hidden_size // num_heads
        vocab_size = cfg.vocab_size
        
        breakdown = {
            'qkv_projection_gflops': (3 * hidden_size * hidden_size * 2 * num_layers) / 1e9,
            'attention_scores_gflops': (num_heads * head_dim * 2 * num_layers) / 1e9,
            'attention_output_gflops': (num_heads * head_dim * 2 * num_layers) / 1e9,
            'output_projection_gflops': (hidden_size * hidden_size * 2 * num_layers) / 1e9,
            'ffn_up_gflops': (hidden_size * intermediate_size * 2 * num_layers) / 1e9,
            'ffn_down_gflops': (intermediate_size * hidden_size * 2 * num_layers) / 1e9,
            'lm_head_gflops': (hidden_size * vocab_size * 2) / 1e9,
            'total_gflops': estimate_flops(model)
        }
        
        logger.debug("FLOPs breakdown:")
        for component, value in breakdown.items():
            if value:
                logger.debug(f"  {component}: {value:.2f} GFLOPs")
        
        return breakdown
        
    except Exception as e:
        logger.warning(f"Could not generate FLOPs breakdown: {e}")
        return None