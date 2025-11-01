"""Latency measurement utilities."""

import time
import logging
from typing import List, Dict
from contextlib import contextmanager

import torch
import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def inference_mode(is_cuda: bool):
    """
    Context manager for optimized inference.
    
    Args:
        is_cuda: Whether using CUDA
    """
    with torch.inference_mode():
        # Only use autocast if CUDA is available AND we're on CUDA
        if is_cuda and torch.cuda.is_available():
            try:
                # Try to use autocast for potential speed improvements
                with torch.cuda.amp.autocast(enabled=True):
                    yield
            except Exception as e:
                # Fall back to no autocast if it fails
                logger.debug(f"Autocast not available or failed: {e}")
                yield
        else:
            yield


def measure_latency(
    model,
    tokenizer,
    device: str,
    is_cuda: bool,
    prompts: List[str],
    num_warmup: int = 3,
    num_runs: int = 10,
    max_new_tokens: int = 128
) -> Dict[str, float]:
    """
    Measure generation latency.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        device: Device to run on
        is_cuda: Whether using CUDA
        prompts: List of prompts to use for benchmarking
        num_warmup: Number of warmup iterations
        num_runs: Number of measurement iterations
        max_new_tokens: Maximum tokens to generate per prompt
        
    Returns:
        Dictionary with latency metrics
    """
    logger.info(f"Measuring latency ({num_warmup} warmup, {num_runs} runs)...")
    
    # Warmup runs
    for _ in range(num_warmup):
        try:
            inputs = tokenizer(prompts[0], return_tensors='pt', padding=True).to(device)
            with inference_mode(is_cuda):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
        except Exception as e:
            logger.warning(f"Warmup run failed: {e}")
    
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    latencies = []
    tokens_generated = []
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        
        try:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(is_cuda):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            
            latencies.append(latency_ms)
            tokens_generated.append(num_tokens)
            
        except Exception as e:
            logger.warning(f"Measurement run {i} failed: {e}")
            continue
    
    if not latencies:
        logger.error("All measurement runs failed")
        return {'ms_per_token': float('inf'), 'avg_tokens_generated': 0.0}
    
    avg_tokens = np.mean(tokens_generated)
    ms_per_token = np.mean(latencies) / avg_tokens if avg_tokens > 0 else float('inf')
    
    logger.info(f"Latency: {ms_per_token:.3f} ms/token")
    return {'ms_per_token': ms_per_token, 'avg_tokens_generated': avg_tokens}


def measure_ttft(
    model,
    tokenizer,
    device: str,
    is_cuda: bool,
    prompt: str,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Measure Time To First Token.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        device: Device to run on
        is_cuda: Whether using CUDA
        prompt: Prompt to use for measurement
        num_runs: Number of measurement iterations
        
    Returns:
        Dictionary with TTFT metrics
    """
    logger.info(f"Measuring TTFT ({num_runs} runs)...")
    
    # Warmup
    inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    for _ in range(2):
        try:
            with inference_mode(is_cuda):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
        except Exception as e:
            logger.warning(f"TTFT warmup failed: {e}")
    
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    ttfts = []
    
    for _ in range(num_runs):
        try:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(is_cuda):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            ttfts.append((end_time - start_time) * 1000)
            
        except Exception as e:
            logger.warning(f"TTFT measurement failed: {e}")
            continue
    
    if not ttfts:
        logger.error("All TTFT measurements failed")
        return {'ttft_ms': float('inf')}
    
    ttft_ms = np.mean(ttfts)
    logger.info(f"TTFT: {ttft_ms:.3f} ms")
    return {'ttft_ms': ttft_ms}
