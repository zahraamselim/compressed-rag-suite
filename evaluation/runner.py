"""Unified runner for running all evaluation benchmarks."""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json

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
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'efficiency': self.efficiency,
            'performance': self.performance,
            'retrieval': self.retrieval
        }
    
    def to_json(self, filepath: str):
        """Save all results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Comprehensive results saved to {filepath}")
    
    def __str__(self) -> str:
        """Pretty print all results."""
        lines = ["\n" + "="*60, "COMPREHENSIVE EVALUATION RESULTS", "="*60]
        
        if self.efficiency:
            lines.append("\n--- EFFICIENCY ---")
            for key, value in self.efficiency.items():
                if isinstance(value, float):
                    lines.append(f"{key:.<40} {value:.4f}")
                else:
                    lines.append(f"{key:.<40} {value}")
        
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
        
        if self.retrieval:
            lines.append("\n--- RETRIEVAL ---")
            for key, value in self.retrieval.items():
                if isinstance(value, float):
                    lines.append(f"{key:.<40} {value:.4f}")
                else:
                    lines.append(f"{key:.<40} {value}")
        
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
        
        # Initialize benchmarks
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
        logger.info("="*60)
        logger.info("Starting comprehensive evaluation")
        logger.info("="*60)
        
        efficiency_kwargs = efficiency_kwargs or {}
        performance_kwargs = performance_kwargs or {}
        retrieval_kwargs = retrieval_kwargs or {}
        
        output_dir = output_dir or self.config.get('evaluation', {}).get('output_dir', './results')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = ComprehensiveResults()
        
        # Run efficiency benchmark
        if run_efficiency:
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
            results.efficiency = efficiency_results.to_dict()
            
            if save_results:
                efficiency_results.to_json(f"{output_dir}/efficiency_results.json")
        
        # Run performance benchmark
        if run_performance:
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
            results.performance = performance_results.to_dict()
            
            if save_results:
                performance_results.to_json(f"{output_dir}/performance_results.json")
        
        # Run retrieval benchmark
        if run_retrieval:
            if self.rag_pipeline is None:
                logger.warning("RAG pipeline not provided, skipping retrieval benchmarks")
            else:
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
                results.retrieval = retrieval_results.to_dict()
                
                if save_results:
                    retrieval_results.to_json(f"{output_dir}/retrieval_results.json")
        
        # Save comprehensive results
        if save_results:
            results.to_json(f"{output_dir}/comprehensive_results.json")
        
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE EVALUATION COMPLETE")
        logger.info("="*60)
        
        if self.verbose:
            print(results)
        
        return results
    
    def run_efficiency_only(self, **kwargs) -> EfficiencyResults:
        """Run only efficiency benchmarks."""
        if self.efficiency_benchmark is None:
            self.efficiency_benchmark = EfficiencyBenchmark(
                model_interface=self.model_interface,
                config=self.config.get('evaluation', {}).get('efficiency', {}),
                verbose=self.verbose
            )
        return self.efficiency_benchmark.run_all(**kwargs)
    
    def run_performance_only(self, **kwargs) -> PerformanceResults:
        """Run only performance benchmarks."""
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
        
        if self.retrieval_benchmark is None:
            self.retrieval_benchmark = RetrievalBenchmark(
                model_interface=self.model_interface,
                rag_pipeline=self.rag_pipeline,
                config=self.config.get('evaluation', {}).get('retrieval', {}),
                verbose=self.verbose
            )
        return self.retrieval_benchmark.run_all(**kwargs)
