"""Efficiency benchmark orchestrator - enhanced with ModelInterface."""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from evaluation.base import ModelBenchmark, BenchmarkResult
from evaluation.efficiency.latency import (
    measure_latency, measure_ttft, measure_prefill_decode_latency
)
from evaluation.efficiency.throughput import measure_throughput, measure_batch_throughput
from evaluation.efficiency.memory import (
    get_peak_memory, get_model_size, get_bits_per_param, 
    get_parameter_count, get_memory_efficiency, reset_memory_stats,
    estimate_kv_cache_size, get_current_memory
)
from evaluation.efficiency.flops import estimate_flops, calculate_mfu
from evaluation.efficiency.energy import estimate_energy
from evaluation.efficiency.device_specs import detect_tdp, detect_peak_tflops

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyResults(BenchmarkResult):
    """Results from efficiency benchmarks."""
    # Device info
    device: str
    device_name: Optional[str] = None
    
    # Latency metrics
    latency_ms_per_token: float = 0.0
    latency_std: float = 0.0
    ttft_ms: float = 0.0
    ttft_std: float = 0.0
    prefill_ms: Optional[float] = None
    decode_ms_per_token: Optional[float] = None
    
    # Throughput metrics
    throughput_tokens_per_sec: float = 0.0
    throughput_std: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    model_size_gb: float = 0.0
    total_params: int = 0
    bits_per_param: Optional[float] = None
    memory_efficiency: Optional[float] = None
    kv_cache_size_mb: Optional[float] = None
    
    # Compute metrics
    flops_per_token_gflops: Optional[float] = None
    mfu_percent: Optional[float] = None
    
    # Energy metrics
    energy_per_token_mj: Optional[float] = None
    tdp_watts: Optional[float] = None
    
    # Comparison metrics (vs baseline)
    compression_ratio: Optional[float] = None
    speedup: Optional[float] = None
    memory_reduction: Optional[float] = None
    
    # Batch throughput (optional)
    batch_throughput: Optional[Dict[int, Dict[str, float]]] = None


class EfficiencyBenchmark(ModelBenchmark[EfficiencyResults]):
    """
    Benchmark suite for measuring model efficiency.
    
    Measures:
        - Latency (ms/token) with statistics
        - Time to First Token (TTFT)
        - Prefill vs Decode latency
        - Throughput (tokens/second)
        - Batch throughput at different sizes
        - Memory usage (peak, model size, KV cache)
        - Model parameters and bits per parameter
        - FLOPs and Model FLOPs Utilization (MFU)
        - Energy consumption estimates
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
        
        # Get device name
        self.device_name = self._get_device_name()
        
        # Hardware specs
        self.tdp_watts = detect_tdp(self.is_cuda)
        self.peak_tflops = detect_peak_tflops(self.is_cuda)
        
        logger.info(f"Efficiency Benchmark initialized")
        logger.info(f"  Device: {self.device} ({self.device_name})")
        logger.info(f"  TDP: {self.tdp_watts}W")
        logger.info(f"  Peak: {self.peak_tflops} TFLOPs")
    
    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.is_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    return torch.cuda.get_device_name(0)
            except:
                pass
        return str(self.device)
    
    def validate_config(self) -> bool:
        """Validate efficiency configuration."""
        if not super().validate_config():
            return False
        
        # Check required fields
        required = ['num_warmup', 'num_runs', 'max_new_tokens', 'prompts']
        missing = [f for f in required if f not in self.config]
        
        if missing:
            logger.warning(f"Config missing fields: {missing}, using defaults")
        
        # Validate prompts
        prompts = self.config.get('prompts', [])
        if not prompts or not isinstance(prompts, list):
            logger.warning("No prompts in config, using defaults")
            self.config['prompts'] = [
                "The capital of France is",
                "Artificial intelligence is defined as",
                "Machine learning models can"
            ]
        
        return True
    
    def run_all(
        self,
        prompts: Optional[List[str]] = None,
        num_warmup: Optional[int] = None,
        num_runs: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        base_results: Optional[Dict[str, Any]] = None,
        measure_batch_throughput: bool = False,
        measure_prefill_decode: bool = True
    ) -> EfficiencyResults:
        """
        Run all efficiency benchmarks.
        
        Args:
            prompts: List of prompts for benchmarking (uses config if None)
            num_warmup: Number of warmup iterations (uses config if None)
            num_runs: Number of measurement iterations (uses config if None)
            max_new_tokens: Maximum tokens to generate (uses config if None)
            base_results: Results from baseline model for comparison
            measure_batch_throughput: Whether to measure batch throughput
            measure_prefill_decode: Whether to measure prefill/decode separately
            
        Returns:
            EfficiencyResults object with all metrics
        """
        # Validate config
        self.validate_config()
        
        # Use config values if not provided
        prompts = prompts or self.config.get('prompts', ["The capital of France is"])
        num_warmup = num_warmup if num_warmup is not None else self.config.get('num_warmup', 3)
        num_runs = num_runs if num_runs is not None else self.config.get('num_runs', 10)
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.get('max_new_tokens', 128)
        
        logger.info("="*60)
        logger.info("Starting efficiency benchmarks")
        logger.info("="*60)
        logger.info(f"Warmup runs: {num_warmup}")
        logger.info(f"Measurement runs: {num_runs}")
        logger.info(f"Max new tokens: {max_new_tokens}")
        logger.info(f"Prompts: {len(prompts)}")
        
        # Reset memory stats if CUDA
        if self.is_cuda:
            reset_memory_stats(self.is_cuda)
        
        # === Static Metrics (Fast) ===
        logger.info("\n--- Static Metrics ---")
        size_gb, _ = get_model_size(self.model)
        param_counts = get_parameter_count(self.model)
        bits_per_param = get_bits_per_param(self.model)
        
        # === Latency Metrics ===
        logger.info("\n--- Latency Metrics ---")
        latency_results = measure_latency(
            self.model_interface,
            prompts, 
            num_warmup, 
            num_runs, 
            max_new_tokens
        )
        
        # === TTFT Metrics ===
        logger.info("\n--- TTFT Metrics ---")
        ttft_results = measure_ttft(
            self.model_interface,
            prompts[0], 
            num_runs
        )
        
        # === Prefill/Decode Metrics (Optional) ===
        prefill_decode_results = None
        if measure_prefill_decode:
            logger.info("\n--- Prefill/Decode Metrics ---")
            try:
                prefill_decode_results = measure_prefill_decode_latency(
                    self.model_interface,
                    prompts[0],
                    num_decode_tokens=max_new_tokens // 2,
                    num_runs=min(5, num_runs)
                )
            except Exception as e:
                logger.warning(f"Prefill/decode measurement failed: {e}")
                prefill_decode_results = None
        
        # === Throughput Metrics ===
        logger.info("\n--- Throughput Metrics ---")
        throughput_results = measure_throughput(
            self.model_interface,
            prompts, 
            num_runs, 
            max_new_tokens
        )
        
        # === Batch Throughput (Optional) ===
        batch_throughput_results = None
        if measure_batch_throughput and len(prompts) >= 4:
            logger.info("\n--- Batch Throughput Metrics ---")
            try:
                batch_throughput_results = measure_batch_throughput(
                    self.model_interface,
                    prompts,
                    batch_sizes=[1, 2, 4, 8],
                    max_new_tokens=max_new_tokens // 2
                )
            except Exception as e:
                logger.warning(f"Batch throughput measurement failed: {e}")
                batch_throughput_results = None
        
        # === Memory Metrics ===
        logger.info("\n--- Memory Metrics ---")
        peak_memory = get_peak_memory(self.is_cuda)
        memory_efficiency = get_memory_efficiency(size_gb, peak_memory)
        
        # Estimate KV cache size
        try:
            kv_cache_size = estimate_kv_cache_size(self.model, batch_size=1, sequence_length=2048)
        except Exception as e:
            logger.warning(f"KV cache estimation failed: {e}")
            kv_cache_size = None
        
        # === Compute Metrics ===
        logger.info("\n--- Compute Metrics ---")
        flops = estimate_flops(self.model)
        throughput = throughput_results['throughput']
        mfu = calculate_mfu(flops, throughput, self.is_cuda, self.peak_tflops) if flops else None
        
        # === Energy Metrics ===
        logger.info("\n--- Energy Metrics ---")
        energy = estimate_energy(latency_results['ms_per_token'], self.tdp_watts)
        
        # === Comparison Metrics ===
        comparison_metrics = {}
        if base_results:
            logger.info("\n--- Comparison Metrics ---")
            comparison_metrics = self._compute_comparison_metrics(
                base_results, size_gb, latency_results['ms_per_token'], peak_memory
            )
        
        # === Create Results ===
        results = EfficiencyResults(
            # Device
            device=str(self.device),
            device_name=self.device_name,
            
            # Latency
            latency_ms_per_token=latency_results['ms_per_token'],
            latency_std=latency_results.get('latency_std', 0.0),
            ttft_ms=ttft_results['ttft_ms'],
            ttft_std=ttft_results.get('ttft_std', 0.0),
            prefill_ms=prefill_decode_results.get('prefill_ms') if prefill_decode_results else None,
            decode_ms_per_token=prefill_decode_results.get('decode_ms_per_token') if prefill_decode_results else None,
            
            # Throughput
            throughput_tokens_per_sec=throughput,
            throughput_std=throughput_results.get('throughput_std', 0.0),
            
            # Memory
            peak_memory_mb=peak_memory,
            model_size_gb=size_gb,
            total_params=param_counts['total'],
            bits_per_param=bits_per_param,
            memory_efficiency=memory_efficiency,
            kv_cache_size_mb=kv_cache_size,
            
            # Compute
            flops_per_token_gflops=flops,
            mfu_percent=mfu,
            
            # Energy
            energy_per_token_mj=energy,
            tdp_watts=self.tdp_watts,
            
            # Comparison
            **comparison_metrics,
            
            # Batch throughput
            batch_throughput=batch_throughput_results
        )
        
        logger.info("\n" + "="*60)
        logger.info("Efficiency benchmarks complete!")
        logger.info("="*60)
        
        if self.verbose:
            print(results)
        
        return results
    
    def _compute_comparison_metrics(
        self,
        base_results: Dict[str, Any],
        size_gb: float,
        latency: float,
        peak_memory: float
    ) -> Dict[str, Optional[float]]:
        """
        Compute comparison metrics against baseline.
        
        Args:
            base_results: Baseline results dictionary
            size_gb: Current model size in GB
            latency: Current latency in ms/token
            peak_memory: Current peak memory in MB
            
        Returns:
            Dictionary with comparison metrics
        """
        metrics = {
            'compression_ratio': None,
            'speedup': None,
            'memory_reduction': None
        }
        
        # Model size compression
        if 'model_size_gb' in base_results and base_results['model_size_gb'] and size_gb > 0:
            metrics['compression_ratio'] = base_results['model_size_gb'] / size_gb
            self._log_metric("Compression ratio", f"{metrics['compression_ratio']:.2f}x")
        
        # Latency speedup
        if 'latency_ms_per_token' in base_results and base_results['latency_ms_per_token'] and latency > 0:
            metrics['speedup'] = base_results['latency_ms_per_token'] / latency
            self._log_metric("Speedup", f"{metrics['speedup']:.2f}x")
        
        # Memory reduction
        if 'peak_memory_mb' in base_results and base_results['peak_memory_mb'] and peak_memory > 0:
            metrics['memory_reduction'] = base_results['peak_memory_mb'] / peak_memory
            self._log_metric("Memory reduction", f"{metrics['memory_reduction']:.2f}x")
        
        return metrics