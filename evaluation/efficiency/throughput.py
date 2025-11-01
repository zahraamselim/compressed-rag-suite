"""Throughput measurement utilities."""

import time
import logging
from typing import List, Dict

import torch

logger = logging.getLogger(__name__)


def measure_throughput(
    model,
    tokenizer,
    device: str,
    is_cuda: bool,
    prompts: List[str],
    num_runs: int = 10,
    max_new_tokens: int = 128
) -> Dict[str, float]:
    """
    Measure generation throughput.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        device: Device to run on
        is_cuda: Whether using CUDA
        prompts: List of prompts to use for benchmarking
        num_runs: Number of measurement iterations
        max_new_tokens: Maximum tokens to generate per prompt
        
    Returns:
        Dictionary with throughput metrics
    """
    logger.info(f"Measuring throughput ({num_runs} runs)...")
    
    total_tokens = 0
    total_time = 0
    
    with torch.no_grad():
        for prompt in prompts[:num_runs]:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            if is_cuda:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            if is_cuda:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            total_tokens += tokens
            total_time += (end_time - start_time)
    
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    logger.info(f"Throughput: {throughput:.2f} tokens/s")
    return {'throughput': throughput}