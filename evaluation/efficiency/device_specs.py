"""Device specification detection utilities."""

import logging
import torch

logger = logging.getLogger(__name__)

GPU_SPECS = {
    't4': {'tdp': 70.0, 'tflops': 8.1},
    'p100': {'tdp': 250.0, 'tflops': 18.7},
    'v100': {'tdp': 300.0, 'tflops': 28.0},
    'a10': {'tdp': 125.0, 'tflops': 31.2},
    'a100': {'tdp': 400.0, 'tflops': 78.0},
    'h100': {'tdp': 700.0, 'tflops': 204.0},
    'rtx 3060': {'tdp': 170.0, 'tflops': 12.7},
    'rtx 3090': {'tdp': 350.0, 'tflops': 35.6},
    'rtx 4090': {'tdp': 450.0, 'tflops': 82.6},
    'rtx 4080': {'tdp': 320.0, 'tflops': 48.7},
    'a6000': {'tdp': 300.0, 'tflops': 38.7},
}


def detect_tdp(is_cuda: bool) -> float:
    """
    Auto-detect TDP based on device.
    
    Args:
        is_cuda: Whether using CUDA
        
    Returns:
        TDP in watts
    """
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0).lower()
        for gpu_key, specs in GPU_SPECS.items():
            if gpu_key in gpu_name:
                logger.info(f"Detected GPU: {gpu_name}")
                return specs['tdp']
        logger.warning(f"Unknown GPU: {gpu_name}, using default TDP: 70W")
        return 70.0
    return 15.0


def detect_peak_tflops(is_cuda: bool) -> float:
    """
    Auto-detect peak TFLOPs based on device.
    
    Args:
        is_cuda: Whether using CUDA
        
    Returns:
        Peak TFLOPs
    """
    if not is_cuda:
        return 0.0
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    for gpu_key, specs in GPU_SPECS.items():
        if gpu_key in gpu_name:
            return specs['tflops']
    
    logger.warning(f"Unknown GPU: {gpu_name}, using default TFLOPs: 8.1")
    return 8.1