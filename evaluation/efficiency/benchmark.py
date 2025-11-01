"""Efficiency benchmark orchestrator - refactored."""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from evaluation.base import ModelBenchmark, BenchmarkResult
from evaluation.efficiency.latency import measure_latency, measure_ttft
from evaluation.efficiency.throughput import measure_throughput
from evaluation.efficiency.memory import get_peak_memory, get_model_size, get_bits_per_param
from evaluation.efficiency.flops import estimate_flops, calculate_mfu
from evaluation.efficiency.energy import estimate_energy
from evaluation.efficiency.device_specs import detect_tdp, detect_peak_tflops

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyResults(BenchmarkResult):
    """Results from efficiency benchmarks."""
    device: str
    latency_ms_per_token: float
    ttft_ms: float
    throughput_tokens_per_sec: float
    peak_memory_mb: float
    model_size_gb: float
    bits_per_param: Optional[float] = None
    flops_per_token_gflops: Optional[float] = None
    mfu_percent: Optional[float] = None
    energy_per_token_mj: Optional[float] = None
    compression_ratio: Optional[float] = None
    speedup: Optional[float] = None
    memory_reduction: Optional[float] = None


class EfficiencyBenchmark(ModelBenchmark[EfficiencyResults]):
    """
    Benchmark suite for measuring model efficiency.
    
    Measures:
        - Latency (ms/token)
        - Time to First Token (TTFT)
        - Throughput (tokens/second)
        - Memory usage
        - Model size and bits per parameter
        - FLOPs and Model FLOPs Utilization (MFU)
        - Energy consumption
    """
    
    def __init__(
        self,
        model_interface,
        config: dict,
        verbose: bool = False
    ):
        """
        Initialize efficiency benchmark.
        
        Args:
            model_interface: ModelInterface instance
            config: Efficiency config from config.json
            verbose: Enable verbose logging
        """
        super().__init__(
            model_interface=model_interface,
            config=config,
            verbose=verbose
        )
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # Device detection
        self.device = model_interface.get_device()
        self.is_cuda = 'cuda' in str(self.device).lower()
        
        # Hardware specs
        self.tdp_watts = detect_tdp(self.is_cuda)
        self.peak_tflops = detect_peak_tflops(self.is_cuda)
        
        logger.info(f"Device: {self.device}")
        logger.info(f"TDP: {self.tdp_watts}W, Peak: {self.peak_tflops} TFLOPs")
    
    def validate_config(self) -> bool:
        """Validate efficiency configuration."""
        if not super().validate_config():
            return False
        
        # Validate config has required fields
        required = ['num_warmup', 'num_runs', 'max_new_tokens']
        for field in required:
            if field not in self.config:
                logger.warning(f"Config missing field: {field}, using default")
        
        return True
    
    def run_all(
        self,
        prompts: Optional[List[str]] = None,
        num_warmup: Optional[int] = None,
        num_runs: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        base_results: Optional[Dict[str, Any]] = None
    ) -> EfficiencyResults:
        """
        Run all efficiency benchmarks.
        
        Args:
            prompts: List of prompts for benchmarking (uses config if None)
            num_warmup: Number of warmup iterations (uses config if None)
            num_runs: Number of measurement iterations (uses config if None)
            max_new_tokens: Maximum tokens to generate (uses config if None)
            base_results: Results from baseline model for comparison
            
        Returns:
            EfficiencyResults object with all metrics
        """
        # Use config values if not provided
        prompts = prompts or self.config.get('prompts', ["The capital of France is"])
        num_warmup = num_warmup if num_warmup is not None else self.config.get('num_warmup', 3)
        num_runs = num_runs if num_runs is not None else self.config.get('num_runs', 10)
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.get('max_new_tokens', 128)
        
        logger.info("Starting efficiency benchmarks...")
        
        # Reset memory stats if CUDA
        if self.is_cuda:
            try:
                import torch
                torch.cuda.reset_peak_memory_stats()
            except ImportError:
                logger.warning("PyTorch not available, skipping CUDA memory reset")
        
        # Run individual benchmarks
        latency_results = self._measure_latency(prompts, num_warmup, num_runs, max_new_tokens)
        ttft_results = self._measure_ttft(prompts[0], num_runs)
        throughput_results = self._measure_throughput(prompts, num_runs, max_new_tokens)
        
        # Static metrics
        size_gb, _ = get_model_size(self.model)
        bits_per_param = get_bits_per_param(self.model)
        peak_memory = get_peak_memory(self.is_cuda)
        
        # Computed metrics
        flops = estimate_flops(self.model)
        throughput = throughput_results['throughput']
        mfu = self._calculate_mfu(flops, throughput) if flops else None
        energy = estimate_energy(latency_results['ms_per_token'], self.tdp_watts)
        
        # Comparison metrics
        comparison_metrics = self._compute_comparison_metrics(
            base_results, size_gb, latency_results['ms_per_token'], peak_memory
        )
        
        results = EfficiencyResults(
            device=str(self.device),
            latency_ms_per_token=latency_results['ms_per_token'],
            ttft_ms=ttft_results['ttft_ms'],
            throughput_tokens_per_sec=throughput,
            peak_memory_mb=peak_memory,
            model_size_gb=size_gb,
            bits_per_param=bits_per_param,
            flops_per_token_gflops=flops,
            mfu_percent=mfu,
            energy_per_token_mj=energy,
            **comparison_metrics
        )
        
        logger.info("Efficiency benchmarks complete!")
        if self.verbose:
            print(results)
        
        return results
    
    def _measure_latency(
        self, prompts: List[str], num_warmup: int, num_runs: int, max_new_tokens: int
    ) -> Dict[str, float]:
        """Measure latency metrics."""
        return measure_latency(
            self.model, self.tokenizer, self.device, self.is_cuda,
            prompts, num_warmup, num_runs, max_new_tokens
        )
    
    def _measure_ttft(self, prompt: str, num_runs: int) -> Dict[str, float]:
        """Measure time to first token."""
        return measure_ttft(
            self.model, self.tokenizer, self.device, self.is_cuda,
            prompt, num_runs
        )
    
    def _measure_throughput(
        self, prompts: List[str], num_runs: int, max_new_tokens: int
    ) -> Dict[str, float]:
        """Measure throughput metrics."""
        return measure_throughput(
            self.model, self.tokenizer, self.device, self.is_cuda,
            prompts, num_runs, max_new_tokens
        )
    
    def _calculate_mfu(self, flops: float, throughput: float) -> Optional[float]:
        """Calculate Model FLOPs Utilization."""
        return calculate_mfu(flops, throughput, self.is_cuda, self.peak_tflops)
    
    def _compute_comparison_metrics(
        self,
        base_results: Optional[Dict[str, Any]],
        size_gb: float,
        latency: float,
        peak_memory: float
    ) -> Dict[str, Optional[float]]:
        """Compute comparison metrics against baseline."""
        metrics = {
            'compression_ratio': None,
            'speedup': None,
            'memory_reduction': None
        }
        
        if not base_results:
            return metrics
        
        logger.info("Computing comparison metrics against baseline...")
        
        if 'model_size_gb' in base_results and base_results['model_size_gb']:
            metrics['compression_ratio'] = base_results['model_size_gb'] / size_gb
            self._log_metric("Compression ratio", f"{metrics['compression_ratio']:.2f}x")
        
        if 'latency_ms_per_token' in base_results and base_results['latency_ms_per_token']:
            metrics['speedup'] = base_results['latency_ms_per_token'] / latency
            self._log_metric("Speedup", f"{metrics['speedup']:.2f}x")
        
        if 'peak_memory_mb' in base_results and base_results['peak_memory_mb'] and peak_memory > 0:
            metrics['memory_reduction'] = base_results['peak_memory_mb'] / peak_memory
            self._log_metric("Memory reduction", f"{metrics['memory_reduction']:.2f}x")
        
        return metrics
