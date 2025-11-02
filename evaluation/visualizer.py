"""Visualization tool for evaluation results."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")

# Check for seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    SEABORN_AVAILABLE = False
    logger.info("seaborn not available (optional). Install with: pip install seaborn")


class ResultsVisualizer:
    """Visualize and compare evaluation results."""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style ('default', 'seaborn', 'dark_background')
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        self.results = []
        self.result_names = []
        self.colors = plt.cm.tab10.colors
        
        if style == 'seaborn' and SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
        elif style != 'default':
            plt.style.use(style)
    
    def load_result(self, filepath: str, name: Optional[str] = None):
        """Load a single result file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {filepath}")
        
        with open(path, 'r') as f:
            result = json.load(f)
        
        result_name = name or path.stem
        self.results.append(result)
        self.result_names.append(result_name)
        
        logger.info(f"Loaded result: {result_name}")
    
    def load_results(self, filepaths: List[str], names: Optional[List[str]] = None):
        """Load multiple result files."""
        if names and len(names) != len(filepaths):
            raise ValueError("Length of names must match filepaths")
        
        for i, filepath in enumerate(filepaths):
            name = names[i] if names else None
            self.load_result(filepath, name)
    
    def _find_metric_value(self, result: Dict, metric: str) -> Optional[float]:
        """Find a metric value in nested dictionary."""
        # Direct key
        if metric in result and isinstance(result[metric], (int, float)):
            return float(result[metric])
        
        # Search in nested dicts
        for key, value in result.items():
            if isinstance(value, dict) and metric in value:
                nested_val = value[metric]
                if isinstance(nested_val, (int, float)):
                    return float(nested_val)
        
        return None
    
    def plot_metric_comparison(
        self,
        metrics: List[str],
        title: str = "Metric Comparison",
        ylabel: str = "Value",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Create bar chart comparing metrics across results.
        
        Args:
            metrics: List of metric names to compare
            title: Plot title
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            save_path: Optional path to save figure
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(metrics))
        width = 0.8 / len(self.results)
        
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            values = []
            for metric in metrics:
                value = self._find_metric_value(result, metric)
                values.append(value if value is not None else 0.0)
            
            offset = (i - len(self.results)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=name, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_efficiency_comparison(
        self,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive efficiency comparison plot.
        
        Shows latency, throughput, memory, and energy metrics.
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        efficiency_metrics = {
            'Latency (ms/tok)': 'latency_ms_per_token',
            'Throughput (tok/s)': 'throughput_tokens_per_sec',
            'Peak Memory (GB)': 'peak_memory_mb',
            'Energy (mJ/tok)': 'energy_per_token_mj'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (label, metric) in enumerate(efficiency_metrics.items()):
            ax = axes[idx]
            
            values = []
            names = []
            for result, name in zip(self.results, self.result_names):
                value = self._find_metric_value(result, metric)
                if value is not None:
                    # Convert memory to GB if needed
                    if metric == 'peak_memory_mb':
                        value = value / 1024
                    values.append(value)
                    names.append(name)
            
            if values:
                colors_subset = [self.colors[i % len(self.colors)] for i in range(len(values))]
                bars = ax.bar(names, values, color=colors_subset, alpha=0.8)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
                
                ax.set_title(label, fontsize=11, fontweight='bold')
                ax.set_ylabel('Value', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Efficiency Comparison', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_comparison(
        self,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Create performance benchmark comparison.
        
        Shows perplexity and accuracy metrics.
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Perplexity
        perplexities = []
        names_with_ppl = []
        for result, name in zip(self.results, self.result_names):
            ppl = self._find_metric_value(result, 'perplexity')
            if ppl is not None:
                perplexities.append(ppl)
                names_with_ppl.append(name)
        
        if perplexities:
            colors_subset = [self.colors[i % len(self.colors)] for i in range(len(perplexities))]
            bars = ax1.bar(names_with_ppl, perplexities, color=colors_subset, alpha=0.8)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
            
            ax1.set_title('Perplexity (lower is better)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Perplexity', fontsize=10)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
        
        # Average accuracy
        accuracies = []
        names_with_acc = []
        for result, name in zip(self.results, self.result_names):
            acc = self._find_metric_value(result, 'average_accuracy')
            if acc is not None:
                accuracies.append(acc * 100)  # Convert to percentage
                names_with_acc.append(name)
        
        if accuracies:
            colors_subset = [self.colors[i % len(self.colors)] for i in range(len(accuracies))]
            bars = ax2.bar(names_with_acc, accuracies, color=colors_subset, alpha=0.8)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
            
            ax2.set_title('Average Accuracy (higher is better)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Accuracy (%)', fontsize=10)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_retrieval_comparison(
        self,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ):
        """Create RAG/retrieval metrics comparison."""
        if not self.results:
            logger.warning("No results loaded")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Precision@K
        k_values = [1, 3, 5, 10]
        ax = axes[0]
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            precisions = []
            for k in k_values:
                p = self._find_metric_value(result, f'precision_at_{k}')
                precisions.append(p if p is not None else 0.0)
            
            ax.plot(k_values, precisions, marker='o', label=name, 
                   color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_title('Precision@K', fontsize=11, fontweight='bold')
        ax.set_xlabel('K', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Answer quality metrics
        ax = axes[1]
        answer_metrics = ['exact_match', 'f1_score', 'faithfulness']
        x = np.arange(len(answer_metrics))
        width = 0.8 / len(self.results)
        
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            values = []
            for metric in answer_metrics:
                value = self._find_metric_value(result, metric)
                values.append(value if value is not None else 0.0)
            
            offset = (i - len(self.results)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=name, 
                  color=self.colors[i % len(self.colors)])
        
        ax.set_title('Answer Quality', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(answer_metrics, rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Context metrics
        ax = axes[2]
        context_metrics = ['context_sufficiency', 'context_precision', 'context_coverage']
        x = np.arange(len(context_metrics))
        
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            values = []
            for metric in context_metrics:
                value = self._find_metric_value(result, metric)
                values.append(value if value is not None else 0.0)
            
            offset = (i - len(self.results)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=name,
                  color=self.colors[i % len(self.colors)])
        
        ax.set_title('Context Quality', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(context_metrics, rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Timing comparison
        ax = axes[3]
        timing_metrics = ['avg_retrieval_time_ms', 'avg_rag_generation_time_ms']
        timing_labels = ['Retrieval', 'Generation']
        x = np.arange(len(timing_labels))
        
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            values = []
            for metric in timing_metrics:
                value = self._find_metric_value(result, metric)
                values.append(value if value is not None else 0.0)
            
            offset = (i - len(self.results)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=name,
                  color=self.colors[i % len(self.colors)])
        
        ax.set_title('RAG Timing', fontsize=11, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(timing_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Retrieval/RAG Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_radar_chart(
        self,
        metrics: List[str],
        title: str = "Performance Radar",
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[str] = None
    ):
        """
        Create radar chart for multi-dimensional comparison.
        
        Args:
            metrics: List of metrics to include (3-8 recommended)
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        if len(metrics) < 3:
            logger.warning("Radar chart needs at least 3 metrics")
            return
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            values = []
            for metric in metrics:
                value = self._find_metric_value(result, metric)
                values.append(value if value is not None else 0.0)
            
            # Normalize values to 0-1 range
            max_vals = [max(self._find_metric_value(r, m) or 0.0 for r in self.results) 
                       for m in metrics]
            normalized = [v / max_v if max_v > 0 else 0 
                         for v, max_v in zip(values, max_vals)]
            normalized += normalized[:1]  # Complete the circle
            
            ax.plot(angles, normalized, 'o-', linewidth=2, label=name,
                   color=self.colors[i % len(self.colors)])
            ax.fill(angles, normalized, alpha=0.15, 
                   color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_dashboard(
        self,
        output_dir: str = './visualizations'
    ):
        """
        Create a complete dashboard with all visualizations.
        
        Args:
            output_dir: Directory to save all plots
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating dashboard in {output_dir}...")
        
        # Efficiency comparison
        try:
            self.plot_efficiency_comparison(
                save_path=str(output_path / 'efficiency_comparison.png')
            )
        except Exception as e:
            logger.error(f"Failed to create efficiency plot: {e}")
        
        # Performance comparison
        try:
            self.plot_performance_comparison(
                save_path=str(output_path / 'performance_comparison.png')
            )
        except Exception as e:
            logger.error(f"Failed to create performance plot: {e}")
        
        # Retrieval comparison
        try:
            self.plot_retrieval_comparison(
                save_path=str(output_path / 'retrieval_comparison.png')
            )
        except Exception as e:
            logger.error(f"Failed to create retrieval plot: {e}")
        
        # Radar chart
        try:
            radar_metrics = [
                'latency_ms_per_token',
                'throughput_tokens_per_sec',
                'average_accuracy',
                'exact_match',
                'f1_score'
            ]
            self.plot_radar_chart(
                metrics=radar_metrics,
                title="Overall Performance Radar",
                save_path=str(output_path / 'radar_comparison.png')
            )
        except Exception as e:
            logger.error(f"Failed to create radar chart: {e}")
        
        logger.info(f"Dashboard created in {output_dir}")


def main():
    """CLI interface for visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('files', nargs='+', help='JSON result files')
    parser.add_argument('--names', nargs='+', help='Custom names for results')
    parser.add_argument('--output-dir', default='./visualizations',
                       help='Output directory for plots')
    parser.add_argument('--dashboard', action='store_true',
                       help='Create complete dashboard')
    parser.add_argument('--efficiency', action='store_true',
                       help='Create efficiency comparison plot')
    parser.add_argument('--performance', action='store_true',
                       help='Create performance comparison plot')
    parser.add_argument('--retrieval', action='store_true',
                       help='Create retrieval comparison plot')
    parser.add_argument('--style', default='default',
                       help='Plot style (default, seaborn, dark_background)')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(style=args.style)
    visualizer.load_results(args.files, args.names)
    
    if args.dashboard:
        visualizer.create_dashboard(args.output_dir)
    else:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if args.efficiency:
            visualizer.plot_efficiency_comparison(
                save_path=str(output_path / 'efficiency_comparison.png')
            )
        if args.performance:
            visualizer.plot_performance_comparison(
                save_path=str(output_path / 'performance_comparison.png')
            )
        if args.retrieval:
            visualizer.plot_retrieval_comparison(
                save_path=str(output_path / 'retrieval_comparison.png')
            )
        
        if not any([args.efficiency, args.performance, args.retrieval]):
            # Default: show all
            visualizer.create_dashboard(args.output_dir)


if __name__ == "__main__":
    main()
