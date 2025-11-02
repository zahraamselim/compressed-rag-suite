"""
Visualization utilities for model evaluation results.
Place in: evaluation/visualizer.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional


class EvaluationVisualizer:
    """Create comprehensive visualizations for model evaluation results."""
    
    @staticmethod
    def create_comprehensive_plot(
        results: Any,
        model_info: Dict[str, Any],
        vram_used: float,
        quantization_type: str,
        save_path: Optional[Path] = None,
        figsize: tuple = (15, 12)
    ):
        """
        Create a comprehensive 2x2 plot with all evaluation metrics.
        
        Args:
            results: EvaluationResults object
            model_info: Dictionary with model information
            vram_used: VRAM usage in GB
            quantization_type: Type of quantization (e.g., 'GPTQ 4-bit')
            save_path: Path to save the figure
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f'Mistral 7B {quantization_type} - Comprehensive Evaluation',
            fontsize=16,
            fontweight='bold'
        )
        
        # 1. Performance benchmarks (top-left)
        EvaluationVisualizer._plot_performance_benchmarks(
            axes[0, 0], 
            results.performance
        )
        
        # 2. RAG comparison (top-right)
        EvaluationVisualizer._plot_rag_metrics(
            axes[0, 1],
            results.retrieval
        )
        
        # 3. Efficiency metrics (bottom-left)
        EvaluationVisualizer._plot_efficiency_metrics(
            axes[1, 0],
            results.efficiency
        )
        
        # 4. Summary statistics (bottom-right)
        EvaluationVisualizer._plot_summary_text(
            axes[1, 1],
            results,
            model_info,
            vram_used,
            quantization_type
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    @staticmethod
    def _plot_performance_benchmarks(ax, performance_results):
        """Plot LM-Eval benchmark scores."""
        if not performance_results or not performance_results.lm_eval_scores:
            ax.text(0.5, 0.5, 'No performance data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('LM-Eval Benchmark Scores')
            return
        
        tasks = list(performance_results.lm_eval_scores.keys())
        scores = [performance_results.lm_eval_scores[t] * 100 for t in tasks]
        
        bars = ax.barh(tasks, scores, color='steelblue', alpha=0.8)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('LM-Eval Benchmark Scores')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 1, i, f'{score:.1f}%', va='center', fontsize=9)
    
    @staticmethod
    def _plot_rag_metrics(ax, retrieval_results):
        """Plot RAG performance metrics."""
        if not retrieval_results:
            ax.text(0.5, 0.5, 'No RAG data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('RAG Performance Metrics')
            return
        
        metrics = ['F1', 'EM', 'Relevance', 'Precision', 'Recall']
        rag_scores = [
            retrieval_results.f1_score,
            retrieval_results.exact_match,
            retrieval_results.answer_relevance,
            retrieval_results.context_precision,
            retrieval_results.context_recall
        ]
        
        x = range(len(metrics))
        ax.bar(x, rag_scores, color='forestgreen', alpha=0.8, label='RAG')
        
        # Add No-RAG comparison if available
        if retrieval_results.no_rag_f1:
            no_rag_scores = [
                retrieval_results.no_rag_f1,
                retrieval_results.no_rag_exact_match
            ] + [0] * 3
            ax.bar([0, 1], no_rag_scores[:2], 
                  color='coral', alpha=0.8, label='No-RAG')
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('RAG Performance Metrics')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
    
    @staticmethod
    def _plot_efficiency_metrics(ax, efficiency_results):
        """Plot efficiency metrics."""
        if not efficiency_results:
            ax.text(0.5, 0.5, 'No efficiency data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Efficiency Metrics')
            return
        
        metrics = ['Latency\n(ms)', 'Throughput\n(tok/s)', 'Memory\n(GB)']
        values = [
            efficiency_results.latency.mean_ms,
            efficiency_results.throughput.tokens_per_second,
            efficiency_results.memory.peak_allocated_gb
        ]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8)
        ax.set_ylabel('Value')
        ax.set_title('Efficiency Metrics')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    @staticmethod
    def _plot_summary_text(ax, results, model_info, vram_used, quantization_type):
        """Plot summary statistics as text."""
        ax.axis('off')
        
        # Build summary text
        summary_lines = [
            "MODEL SUMMARY",
            "=" * 40,
            "",
            f"Quantization: {quantization_type}",
            f"Model Size: {model_info['size_gb']:.2f} GB",
            f"VRAM Usage: {vram_used:.2f} GB",
            f"Parameters: {model_info['num_parameters']:,}",
            "",
            "PERFORMANCE",
            "=" * 40,
        ]
        
        if results.performance and results.performance.average_accuracy:
            summary_lines.append(
                f"Avg Accuracy: {results.performance.average_accuracy*100:.2f}%"
            )
        
        summary_lines.extend([
            "",
            "EFFICIENCY",
            "=" * 40,
        ])
        
        if results.efficiency:
            summary_lines.extend([
                f"Latency: {results.efficiency.latency.mean_ms:.2f}ms",
                f"Throughput: {results.efficiency.throughput.tokens_per_second:.2f} tok/s",
            ])
        
        summary_lines.extend([
            "",
            "RAG PERFORMANCE",
            "=" * 40,
        ])
        
        if results.retrieval:
            summary_lines.append(f"F1 Score: {results.retrieval.f1_score:.4f}")
            if results.retrieval.no_rag_f1:
                improvement = (results.retrieval.f1_score - 
                             results.retrieval.no_rag_f1) * 100
                summary_lines.append(f"RAG Improvement: {improvement:+.2f}%")
        
        summary_text = "\n".join(summary_lines)
        
        ax.text(0.1, 0.9, summary_text, 
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    @staticmethod
    def create_comparison_plot(
        results_dict: Dict[str, Any],
        metric: str = 'f1_score',
        save_path: Optional[Path] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Create comparison plot across multiple quantization methods.
        
        Args:
            results_dict: Dict mapping model names to results objects
            metric: Metric to compare ('f1_score', 'accuracy', 'latency', etc.)
            save_path: Path to save the figure
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        model_names = list(results_dict.keys())
        values = []
        
        for name, results in results_dict.items():
            if metric == 'f1_score' and results.retrieval:
                values.append(results.retrieval.f1_score)
            elif metric == 'accuracy' and results.performance:
                values.append(results.performance.average_accuracy)
            elif metric == 'latency' and results.efficiency:
                values.append(results.efficiency.latency.mean_ms)
            elif metric == 'throughput' and results.efficiency:
                values.append(results.efficiency.throughput.tokens_per_second)
            else:
                values.append(0)
        
        bars = ax.bar(model_names, values, alpha=0.8)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Quantization Method Comparison - {metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}' if value < 10 else f'{value:.2f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_efficiency_comparison(
        results_dict: Dict[str, Any],
        save_path: Optional[Path] = None,
        figsize: tuple = (14, 8)
    ):
        """Create comprehensive efficiency comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Efficiency Metrics Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(results_dict.keys())
        
        # Latency
        latencies = [r.efficiency.latency.mean_ms if r.efficiency else 0 
                    for r in results_dict.values()]
        axes[0, 0].bar(model_names, latencies, color='skyblue', alpha=0.8)
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].set_title('Inference Latency')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Throughput
        throughputs = [r.efficiency.throughput.tokens_per_second if r.efficiency else 0
                      for r in results_dict.values()]
        axes[0, 1].bar(model_names, throughputs, color='lightcoral', alpha=0.8)
        axes[0, 1].set_ylabel('Tokens/second')
        axes[0, 1].set_title('Throughput')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Memory
        memories = [r.efficiency.memory.peak_allocated_gb if r.efficiency else 0
                   for r in results_dict.values()]
        axes[1, 0].bar(model_names, memories, color='lightgreen', alpha=0.8)
        axes[1, 0].set_ylabel('Memory (GB)')
        axes[1, 0].set_title('Peak Memory Usage')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Combined radar plot (if more than 2 models)
        if len(model_names) >= 2:
            EvaluationVisualizer._plot_efficiency_radar(
                axes[1, 1], model_names, results_dict
            )
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def _plot_efficiency_radar(ax, model_names, results_dict):
        """Create radar plot for efficiency metrics."""
        import numpy as np
        
        categories = ['Latency\n(lower better)', 'Throughput', 'Memory\n(lower better)']
        N = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 4, projection='polar')
        
        for name, results in results_dict.items():
            if not results.efficiency:
                continue
            
            # Normalize values (0-1, where 1 is best)
            lat_norm = 1 - (results.efficiency.latency.mean_ms / 100)  # Assume 100ms baseline
            thr_norm = results.efficiency.throughput.tokens_per_second / 100  # Assume 100 tok/s baseline
            mem_norm = 1 - (results.efficiency.memory.peak_allocated_gb / 10)  # Assume 10GB baseline
            
            values = [max(0, min(1, lat_norm)), 
                     max(0, min(1, thr_norm)),
                     max(0, min(1, mem_norm))]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.set_title('Efficiency Profile', size=10, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)


# Example usage:
"""
from evaluation.visualizer import EvaluationVisualizer

# Single model visualization
fig = EvaluationVisualizer.create_comprehensive_plot(
    results=results,
    model_info=info,
    vram_used=vram_used,
    quantization_type='GPTQ 4-bit',
    save_path=RESULTS_DIR / 'evaluation_summary.png'
)
plt.show()

# Multi-model comparison
results_dict = {
    'GPTQ': gptq_results,
    'AWQ': awq_results,
    'HQQ': hqq_results,
    'NF4': nf4_results
}

fig = EvaluationVisualizer.create_comparison_plot(
    results_dict=results_dict,
    metric='f1_score',
    save_path=Path('./comparison_f1.png')
)

fig = EvaluationVisualizer.create_efficiency_comparison(
    results_dict=results_dict,
    save_path=Path('./efficiency_comparison.png')
)
"""
