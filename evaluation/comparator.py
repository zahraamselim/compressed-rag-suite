"""Statistical comparison tool for evaluation results."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for scipy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install with: pip install scipy")


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two models."""
    metric: str
    baseline_value: float
    comparison_value: float
    absolute_diff: float
    relative_change_pct: float
    improved: bool
    statistically_significant: bool = False
    p_value: Optional[float] = None
    effect_size: Optional[float] = None


class ResultsComparator:
    """Statistical comparison between evaluation results."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize comparator.
        
        Args:
            significance_level: P-value threshold for significance (default: 0.05)
        """
        self.significance_level = significance_level
        self.results = []
        self.result_names = []
    
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
    
    def compare_two(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: Optional[List[str]] = None
    ) -> List[ComparisonResult]:
        """
        Compare two results statistically.
        
        Args:
            baseline_idx: Index of baseline result
            comparison_idx: Index of comparison result
            metrics: List of metrics to compare (None = auto-detect)
            
        Returns:
            List of ComparisonResult objects
        """
        if baseline_idx >= len(self.results) or comparison_idx >= len(self.results):
            raise IndexError("Invalid result index")
        
        baseline = self.results[baseline_idx]
        comparison = self.results[comparison_idx]
        
        # Auto-detect metrics if not provided
        if metrics is None:
            metrics = self._detect_numeric_metrics(baseline, comparison)
        
        comparisons = []
        
        for metric in metrics:
            baseline_val = self._find_metric_value(baseline, metric)
            comparison_val = self._find_metric_value(comparison, metric)
            
            if baseline_val is None or comparison_val is None:
                logger.warning(f"Metric '{metric}' not found in both results")
                continue
            
            # Calculate differences
            abs_diff = comparison_val - baseline_val
            rel_change = (abs_diff / baseline_val * 100) if baseline_val != 0 else float('inf')
            
            # Determine if improvement
            is_higher_better = self._is_higher_better(metric)
            improved = (abs_diff > 0) if is_higher_better else (abs_diff < 0)
            
            comparison_result = ComparisonResult(
                metric=metric,
                baseline_value=baseline_val,
                comparison_value=comparison_val,
                absolute_diff=abs_diff,
                relative_change_pct=rel_change,
                improved=improved
            )
            
            comparisons.append(comparison_result)
        
        return comparisons
    
    def _detect_numeric_metrics(
        self,
        result1: Dict,
        result2: Dict
    ) -> List[str]:
        """Detect common numeric metrics between two results."""
        metrics = set()
        
        def extract_numeric_keys(d: Dict, prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (int, float)):
                    metrics.add(full_key)
                elif isinstance(value, dict):
                    extract_numeric_keys(value, full_key)
        
        extract_numeric_keys(result1)
        
        # Only keep metrics that exist in both
        result2_metrics = set()
        extract_numeric_keys(result2)
        
        return sorted(list(metrics & result2_metrics))
    
    def _is_higher_better(self, metric: str) -> bool:
        """Determine if higher values are better for a metric."""
        # Metrics where higher is better
        higher_better = ['accuracy', 'precision', 'recall', 'f1', 'throughput', 
                        'mfu', 'score', 'map', 'mrr', 'tokens_per_sec', 'speedup',
                        'sufficiency', 'coverage', 'faithfulness', 'rouge', 'bertscore',
                        'improvement', 'exact_match']
        
        # Metrics where lower is better
        lower_better = ['latency', 'perplexity', 'loss', 'memory', 'energy', 
                       'ms_per_token', 'time', 'error', 'std']
        
        metric_lower = metric.lower()
        
        # Check if any higher_better keyword is in metric name
        if any(hb in metric_lower for hb in higher_better):
            return True
        
        # Check if any lower_better keyword is in metric name
        if any(lb in metric_lower for lb in lower_better):
            return False
        
        # Default to higher is better
        return True
    
    def print_comparison(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: Optional[List[str]] = None,
        show_all: bool = False
    ):
        """
        Print formatted comparison results.
        
        Args:
            baseline_idx: Index of baseline result
            comparison_idx: Index of comparison result
            metrics: List of metrics to compare
            show_all: Show all metrics (False = only improvements)
        """
        comparisons = self.compare_two(baseline_idx, comparison_idx, metrics)
        
        baseline_name = self.result_names[baseline_idx]
        comparison_name = self.result_names[comparison_idx]
        
        print("\n" + "="*80)
        print(f"COMPARISON: {comparison_name} vs {baseline_name} (baseline)")
        print("="*80)
        
        improvements = [c for c in comparisons if c.improved]
        regressions = [c for c in comparisons if not c.improved and c.absolute_diff != 0]
        unchanged = [c for c in comparisons if c.absolute_diff == 0]
        
        print(f"\nSummary: {len(improvements)} improvements, {len(regressions)} regressions, {len(unchanged)} unchanged")
        
        if improvements:
            print(f"\n{'─'*80}")
            print("✓ IMPROVEMENTS")
            print(f"{'─'*80}")
            for comp in sorted(improvements, key=lambda x: abs(x.relative_change_pct), reverse=True):
                self._print_comparison_line(comp)
        
        if regressions:
            print(f"\n{'─'*80}")
            print("✗ REGRESSIONS")
            print(f"{'─'*80}")
            for comp in sorted(regressions, key=lambda x: abs(x.relative_change_pct), reverse=True):
                self._print_comparison_line(comp)
        
        if show_all and unchanged:
            print(f"\n{'─'*80}")
            print("= UNCHANGED")
            print(f"{'─'*80}")
            for comp in unchanged:
                self._print_comparison_line(comp)
    
    def _print_comparison_line(self, comp: ComparisonResult):
        """Print a single comparison line."""
        symbol = "↑" if comp.improved else "↓"
        
        # Format the change
        if abs(comp.relative_change_pct) < 0.01:
            change_str = f"({comp.absolute_diff:+.4f})"
        elif abs(comp.relative_change_pct) > 1000:
            change_str = f"({comp.absolute_diff:+.2f})"
        else:
            change_str = f"({comp.relative_change_pct:+.2f}%)"
        
        print(f"  {symbol} {comp.metric:<35} "
              f"{comp.baseline_value:>10.4f} → {comp.comparison_value:>10.4f} "
              f"{change_str:>12}")
    
    def get_summary_statistics(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for comparison.
        
        Returns:
            Dictionary with summary stats
        """
        comparisons = self.compare_two(baseline_idx, comparison_idx, metrics)
        
        improvements = [c for c in comparisons if c.improved]
        regressions = [c for c in comparisons if not c.improved and c.absolute_diff != 0]
        
        # Calculate average improvements
        if improvements:
            avg_improvement = np.mean([abs(c.relative_change_pct) for c in improvements])
            max_improvement = max(improvements, key=lambda x: abs(x.relative_change_pct))
        else:
            avg_improvement = 0.0
            max_improvement = None
        
        if regressions:
            avg_regression = np.mean([abs(c.relative_change_pct) for c in regressions])
            max_regression = max(regressions, key=lambda x: abs(x.relative_change_pct))
        else:
            avg_regression = 0.0
            max_regression = None
        
        return {
            'num_metrics': len(comparisons),
            'num_improvements': len(improvements),
            'num_regressions': len(regressions),
            'avg_improvement_pct': avg_improvement,
            'avg_regression_pct': avg_regression,
            'max_improvement': {
                'metric': max_improvement.metric,
                'change_pct': max_improvement.relative_change_pct
            } if max_improvement else None,
            'max_regression': {
                'metric': max_regression.metric,
                'change_pct': max_regression.relative_change_pct
            } if max_regression else None
        }
    
    def compare_all_pairs(
        self,
        metrics: Optional[List[str]] = None
    ) -> Dict[Tuple[int, int], List[ComparisonResult]]:
        """
        Compare all pairs of results.
        
        Returns:
            Dictionary mapping (baseline_idx, comparison_idx) to comparison results
        """
        if len(self.results) < 2:
            logger.warning("Need at least 2 results for pairwise comparison")
            return {}
        
        comparisons = {}
        
        for i in range(len(self.results)):
            for j in range(i + 1, len(self.results)):
                comparisons[(i, j)] = self.compare_two(i, j, metrics)
        
        return comparisons
    
    def find_best_model(
        self,
        metric: str,
        higher_is_better: Optional[bool] = None
    ) -> Tuple[int, float, str]:
        """
        Find the best performing model for a metric.
        
        Args:
            metric: Metric to optimize
            higher_is_better: Whether higher is better (auto-detect if None)
            
        Returns:
            (index, value, name) of best model
        """
        if higher_is_better is None:
            higher_is_better = self._is_higher_better(metric)
        
        best_idx = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for i, result in enumerate(self.results):
            value = self._find_metric_value(result, metric)
            if value is None:
                continue
            
            if higher_is_better:
                if value > best_value:
                    best_value = value
                    best_idx = i
            else:
                if value < best_value:
                    best_value = value
                    best_idx = i
        
        if best_idx is None:
            raise ValueError(f"Metric '{metric}' not found in any results")
        
        return best_idx, best_value, self.result_names[best_idx]
    
    def create_leaderboard(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Create a weighted leaderboard across multiple metrics.
        
        Args:
            metrics: List of metrics to include
            weights: Optional weights for each metric (equal if None)
            
        Returns:
            List of (name, score) tuples sorted by score
        """
        if weights is None:
            weights = [1.0] * len(metrics)
        
        if len(weights) != len(metrics):
            raise ValueError("Number of weights must match number of metrics")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        scores = []
        
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            score = 0.0
            valid_metrics = 0
            
            for metric, weight in zip(metrics, weights):
                value = self._find_metric_value(result, metric)
                if value is None:
                    continue
                
                # Normalize to 0-1 range based on all results
                all_values = [self._find_metric_value(r, metric) for r in self.results]
                all_values = [v for v in all_values if v is not None]
                
                if not all_values:
                    continue
                
                min_val = min(all_values)
                max_val = max(all_values)
                
                if max_val == min_val:
                    normalized = 1.0
                else:
                    normalized = (value - min_val) / (max_val - min_val)
                    
                    # Invert if lower is better
                    if not self._is_higher_better(metric):
                        normalized = 1.0 - normalized
                
                score += normalized * weight
                valid_metrics += 1
            
            if valid_metrics > 0:
                scores.append((name, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def print_leaderboard(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None
    ):
        """Print formatted leaderboard."""
        leaderboard = self.create_leaderboard(metrics, weights)
        
        print("\n" + "="*80)
        print("LEADERBOARD")
        print("="*80)
        print(f"\nMetrics: {', '.join(metrics)}")
        if weights:
            print(f"Weights: {', '.join(f'{w:.2f}' for w in weights)}")
        print()
        
        for rank, (name, score) in enumerate(leaderboard, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
            print(f"{medal} {rank}. {name:<40} Score: {score:.4f}")


def main():
    """CLI interface for comparator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare evaluation results')
    parser.add_argument('files', nargs='+', help='JSON result files')
    parser.add_argument('--names', nargs='+', help='Custom names for results')
    parser.add_argument('--baseline', type=int, default=0,
                       help='Index of baseline result (default: 0)')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to compare')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all metrics including unchanged')
    parser.add_argument('--leaderboard', nargs='+',
                       help='Create leaderboard for these metrics')
    parser.add_argument('--weights', nargs='+', type=float,
                       help='Weights for leaderboard metrics')
    parser.add_argument('--best', help='Find best model for this metric')
    
    args = parser.parse_args()
    
    comparator = ResultsComparator()
    comparator.load_results(args.files, args.names)
    
    if args.leaderboard:
        comparator.print_leaderboard(args.leaderboard, args.weights)
    elif args.best:
        idx, value, name = comparator.find_best_model(args.best)
        print(f"\nBest model for '{args.best}': {name}")
        print(f"Value: {value:.4f}")
    else:
        # Compare against baseline
        for i in range(len(comparator.results)):
            if i != args.baseline:
                comparator.print_comparison(
                    args.baseline, i, args.metrics, args.show_all
                )
                print()


if __name__ == "__main__":
    main()
