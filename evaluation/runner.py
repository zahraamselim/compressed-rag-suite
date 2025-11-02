"""Unified runner for running all evaluation benchmarks."""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.getLogger(__name__).warning("tqdm not available. Progress bars disabled.")

from evaluation.efficiency import EfficiencyBenchmark, EfficiencyResults
from evaluation.performance import PerformanceBenchmark, PerformanceResults
from evaluation.retrieval import RetrievalBenchmark, RetrievalResults

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveResults:
    """Combined results from all benchmarks."""
    efficiency: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None
    timing: Optional[Dict[str, float]] = None
    errors: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'efficiency': self.efficiency,
            'performance': self.performance,
            'retrieval': self.retrieval,
            'timing': self.timing,
            'errors': self.errors,
            'metadata': self.metadata
        }
    
    def to_json(self, filepath: str):
        """Save all results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Comprehensive results saved to {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get high-level summary of key metrics.
        
        Returns:
            Dict with most important metrics from each category
        """
        summary = {}
        
        # Efficiency metrics
        if self.efficiency:
            summary['latency_ms_per_token'] = self.efficiency.get('latency_ms_per_token')
            summary['throughput_tokens_per_sec'] = self.efficiency.get('throughput_tokens_per_sec')
            
            # Convert memory to GB if in MB
            peak_memory_mb = self.efficiency.get('peak_memory_mb')
            if peak_memory_mb:
                summary['peak_memory_gb'] = peak_memory_mb / 1024
            
            summary['model_size_gb'] = self.efficiency.get('model_size_gb')
        
        # Performance metrics
        if self.performance:
            summary['perplexity'] = self.performance.get('perplexity')
            summary['average_accuracy'] = self.performance.get('average_accuracy')
            
            # Get best performing task
            lm_eval_scores = self.performance.get('lm_eval_scores', {})
            if lm_eval_scores:
                best_task = max(lm_eval_scores.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                summary['best_task'] = {best_task[0]: best_task[1]}
        
        # Retrieval metrics
        if self.retrieval:
            precision_at_k = self.retrieval.get('precision_at_k', {})
            if precision_at_k:
                summary['precision_at_3'] = precision_at_k.get(3)
            
            summary['exact_match'] = self.retrieval.get('exact_match')
            summary['faithfulness'] = self.retrieval.get('faithfulness')
        
        # Timing
        if self.timing:
            summary['total_eval_time_seconds'] = self.timing.get('total_seconds')
        
        # Remove None values
        return {k: v for k, v in summary.items() if v is not None}
    
    def __str__(self) -> str:
        """Pretty print all results."""
        lines = ["\n" + "="*60, "COMPREHENSIVE EVALUATION RESULTS", "="*60]
        
        # Efficiency
        if self.efficiency:
            lines.append("\n--- EFFICIENCY ---")
            for key, value in self.efficiency.items():
                if isinstance(value, float):
                    lines.append(f"{key:.<40} {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            lines.append(f"  {k:.<38} {v:.4f}")
                        else:
                            lines.append(f"  {k:.<38} {v}")
                else:
                    lines.append(f"{key:.<40} {value}")
        
        # Performance
        if self.performance:
            lines.append("\n--- PERFORMANCE ---")
            for key, value in self.performance.items():
                if isinstance(value, float):
                    lines.append(f"{key:.<40} {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            lines.append(f"  {k:.<38} {v:.4f}")
                        else:
                            lines.append(f"  {k:.<38} {v}")
                else:
                    lines.append(f"{key:.<40} {value}")
        
        # Retrieval
        if self.retrieval:
            lines.append("\n--- RETRIEVAL ---")
            for key, value in self.retrieval.items():
                if isinstance(value, float):
                    lines.append(f"{key:.<40} {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            lines.append(f"  {k:.<38} {v:.4f}")
                        else:
                            lines.append(f"  {k:.<38} {v}")
                else:
                    lines.append(f"{key:.<40} {value}")
        
        # Timing
        if self.timing:
            lines.append("\n--- TIMING ---")
            for key, value in self.timing.items():
                if isinstance(value, float):
                    lines.append(f"{key:.<40} {value:.2f}s")
                else:
                    lines.append(f"{key:.<40} {value}")
        
        # Errors
        if self.errors:
            lines.append("\n--- ERRORS ---")
            for benchmark, error in self.errors.items():
                lines.append(f"{benchmark}: {error}")
        
        lines.append("="*60)
        return '\n'.join(lines)


class EvaluationRunner:
    """
    Unified runner for running all evaluation benchmarks.
    
    Runs efficiency, performance, and retrieval benchmarks in sequence
    and aggregates results.
    """
    
    def __init__(
        self,
        model_interface,
        config: dict,
        rag_pipeline=None,
        verbose: bool = False
    ):
        """
        Initialize evaluation runner.
        
        Args:
            model_interface: ModelInterface instance
            config: Full config dict with all evaluation settings
            rag_pipeline: RAGPipeline instance (required for retrieval eval)
            verbose: Enable verbose logging
        """
        self.model_interface = model_interface
        self.config = config
        self.rag_pipeline = rag_pipeline
        self.verbose = verbose
        
        # Initialize benchmarks (lazy initialization)
        self.efficiency_benchmark = None
        self.performance_benchmark = None
        self.retrieval_benchmark = None
        
        logger.info("Evaluation runner initialized")
    
    def run_all(
        self,
        run_efficiency: bool = True,
        run_performance: bool = True,
        run_retrieval: bool = True,
        efficiency_kwargs: Optional[Dict[str, Any]] = None,
        performance_kwargs: Optional[Dict[str, Any]] = None,
        retrieval_kwargs: Optional[Dict[str, Any]] = None,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> ComprehensiveResults:
        """
        Run all enabled benchmarks.
        
        Args:
            run_efficiency: Whether to run efficiency benchmarks
            run_performance: Whether to run performance benchmarks
            run_retrieval: Whether to run retrieval benchmarks
            efficiency_kwargs: Additional kwargs for efficiency benchmark
            performance_kwargs: Additional kwargs for performance benchmark
            retrieval_kwargs: Additional kwargs for retrieval benchmark
            save_results: Whether to save results to disk
            output_dir: Output directory for results (uses config if None)
            
        Returns:
            ComprehensiveResults object with all benchmark results
        """
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("Starting comprehensive evaluation")
        logger.info("="*60)
        
        efficiency_kwargs = efficiency_kwargs or {}
        performance_kwargs = performance_kwargs or {}
        retrieval_kwargs = retrieval_kwargs or {}
        
        output_dir = output_dir or self.config.get('evaluation', {}).get('output_dir', './results')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = ComprehensiveResults()
        results.timing = {}
        results.errors = {}
        results.metadata = {
            'model_path': self.model_interface.model_path,
            'model_type': self.model_interface.model_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Determine which benchmarks to run
        benchmarks_to_run = []
        if run_efficiency:
            benchmarks_to_run.append('efficiency')
        if run_performance:
            benchmarks_to_run.append('performance')
        if run_retrieval and self.rag_pipeline is not None:
            benchmarks_to_run.append('retrieval')
        
        # Progress bar setup
        if TQDM_AVAILABLE:
            pbar = tqdm(total=len(benchmarks_to_run), desc="Running Benchmarks", 
                       unit="benchmark", ncols=80)
        else:
            pbar = None
        
        # Run efficiency benchmark
        if run_efficiency:
            benchmark_start = time.time()
            try:
                if pbar:
                    pbar.set_description("Efficiency")
                
                logger.info("\n" + "="*60)
                logger.info("RUNNING EFFICIENCY BENCHMARKS")
                logger.info("="*60)
                
                if self.efficiency_benchmark is None:
                    self.efficiency_benchmark = EfficiencyBenchmark(
                        model_interface=self.model_interface,
                        config=self.config.get('evaluation', {}).get('efficiency', {}),
                        verbose=self.verbose
                    )
                
                efficiency_results = self.efficiency_benchmark.run_all(**efficiency_kwargs)
                
                # Validate results
                if efficiency_results.validate():
                    results.efficiency = efficiency_results.to_dict()
                    
                    if save_results:
                        efficiency_results.to_json(f"{output_dir}/efficiency_results.json")
                else:
                    logger.warning("Efficiency results validation failed")
                    results.errors['efficiency'] = "Validation failed"
                
                results.timing['efficiency_seconds'] = time.time() - benchmark_start
                logger.info(f"Efficiency benchmark completed in {results.timing['efficiency_seconds']:.2f}s")
                
            except Exception as e:
                logger.error(f"Efficiency benchmark failed: {e}", exc_info=self.verbose)
                results.errors['efficiency'] = str(e)
                results.timing['efficiency_seconds'] = time.time() - benchmark_start
            
            finally:
                if pbar:
                    pbar.update(1)
        
        # Run performance benchmark
        if run_performance:
            benchmark_start = time.time()
            try:
                if pbar:
                    pbar.set_description("Performance")
                
                logger.info("\n" + "="*60)
                logger.info("RUNNING PERFORMANCE BENCHMARKS")
                logger.info("="*60)
                
                if self.performance_benchmark is None:
                    self.performance_benchmark = PerformanceBenchmark(
                        model_interface=self.model_interface,
                        config=self.config.get('evaluation', {}).get('performance', {}),
                        verbose=self.verbose
                    )
                
                performance_results = self.performance_benchmark.run_all(**performance_kwargs)
                
                # Validate results
                if performance_results.validate():
                    results.performance = performance_results.to_dict()
                    
                    if save_results:
                        performance_results.to_json(f"{output_dir}/performance_results.json")
                else:
                    logger.warning("Performance results validation failed")
                    results.errors['performance'] = "Validation failed"
                
                results.timing['performance_seconds'] = time.time() - benchmark_start
                logger.info(f"Performance benchmark completed in {results.timing['performance_seconds']:.2f}s")
                
            except Exception as e:
                logger.error(f"Performance benchmark failed: {e}", exc_info=self.verbose)
                results.errors['performance'] = str(e)
                results.timing['performance_seconds'] = time.time() - benchmark_start
            
            finally:
                if pbar:
                    pbar.update(1)
        
        # Run retrieval benchmark
        if run_retrieval:
            if self.rag_pipeline is None:
                logger.warning("RAG pipeline not provided, skipping retrieval benchmarks")
                results.errors['retrieval'] = "RAG pipeline not provided"
            else:
                benchmark_start = time.time()
                try:
                    if pbar:
                        pbar.set_description("Retrieval")
                    
                    logger.info("\n" + "="*60)
                    logger.info("RUNNING RETRIEVAL BENCHMARKS")
                    logger.info("="*60)
                    
                    if self.retrieval_benchmark is None:
                        self.retrieval_benchmark = RetrievalBenchmark(
                            model_interface=self.model_interface,
                            rag_pipeline=self.rag_pipeline,
                            config=self.config.get('evaluation', {}).get('retrieval', {}),
                            verbose=self.verbose
                        )
                    
                    retrieval_results = self.retrieval_benchmark.run_all(**retrieval_kwargs)
                    
                    # Validate results
                    if retrieval_results.validate():
                        results.retrieval = retrieval_results.to_dict()
                        
                        if save_results:
                            retrieval_results.to_json(f"{output_dir}/retrieval_results.json")
                    else:
                        logger.warning("Retrieval results validation failed")
                        results.errors['retrieval'] = "Validation failed"
                    
                    results.timing['retrieval_seconds'] = time.time() - benchmark_start
                    logger.info(f"Retrieval benchmark completed in {results.timing['retrieval_seconds']:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Retrieval benchmark failed: {e}", exc_info=self.verbose)
                    results.errors['retrieval'] = str(e)
                    results.timing['retrieval_seconds'] = time.time() - benchmark_start
                
                finally:
                    if pbar:
                        pbar.update(1)
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Calculate total time
        results.timing['total_seconds'] = time.time() - start_time
        
        # Save comprehensive results
        if save_results:
            results.to_json(f"{output_dir}/comprehensive_results.json")
            
            # Also save summary
            summary = results.get_summary()
            with open(f"{output_dir}/summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to {output_dir}/summary.json")
        
        # Log any errors
        if results.errors:
            logger.warning(f"Evaluation completed with errors in: {list(results.errors.keys())}")
        
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE EVALUATION COMPLETE")
        logger.info(f"Total time: {results.timing['total_seconds']:.2f}s")
        logger.info("="*60)
        
        if self.verbose:
            print(results)
            print("\n--- SUMMARY ---")
            for key, value in results.get_summary().items():
                if isinstance(value, float):
                    print(f"{key:.<40} {value:.4f}")
                else:
                    print(f"{key:.<40} {value}")
        
        return results
    
    def run_efficiency_only(self, **kwargs) -> EfficiencyResults:
        """Run only efficiency benchmarks."""
        logger.info("Running efficiency benchmarks only")
        
        if self.efficiency_benchmark is None:
            self.efficiency_benchmark = EfficiencyBenchmark(
                model_interface=self.model_interface,
                config=self.config.get('evaluation', {}).get('efficiency', {}),
                verbose=self.verbose
            )
        
        return self.efficiency_benchmark.run_all(**kwargs)
    
    def run_performance_only(self, **kwargs) -> PerformanceResults:
        """Run only performance benchmarks."""
        logger.info("Running performance benchmarks only")
        
        if self.performance_benchmark is None:
            self.performance_benchmark = PerformanceBenchmark(
                model_interface=self.model_interface,
                config=self.config.get('evaluation', {}).get('performance', {}),
                verbose=self.verbose
            )
        
        return self.performance_benchmark.run_all(**kwargs)
    
    def run_retrieval_only(self, **kwargs) -> RetrievalResults:
        """Run only retrieval benchmarks."""
        if self.rag_pipeline is None:
            raise ValueError("RAG pipeline required for retrieval benchmarks")
        
        logger.info("Running retrieval benchmarks only")
        
        if self.retrieval_benchmark is None:
            self.retrieval_benchmark = RetrievalBenchmark(
                model_interface=self.model_interface,
                rag_pipeline=self.rag_pipeline,
                config=self.config.get('evaluation', {}).get('retrieval', {}),
                verbose=self.verbose
            )
        
        return self.retrieval_benchmark.run_all(**kwargs)