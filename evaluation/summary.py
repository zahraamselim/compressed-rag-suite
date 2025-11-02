"""Summary tool for evaluation results."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResultsSummary:
    """Pretty-print evaluation results from JSON files."""
    
    def __init__(self):
        self.results = []
        self.result_names = []
    
    def load_result(self, filepath: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a single result file.
        
        Args:
            filepath: Path to JSON file
            name: Optional name for this result (uses filename if None)
            
        Returns:
            Loaded result dictionary
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {filepath}")
        
        with open(path, 'r') as f:
            result = json.load(f)
        
        result_name = name or path.stem
        self.results.append(result)
        self.result_names.append(result_name)
        
        logger.info(f"Loaded result: {result_name}")
        return result
    
    def load_results(self, filepaths: List[str], names: Optional[List[str]] = None):
        """
        Load multiple result files.
        
        Args:
            filepaths: List of paths to JSON files
            names: Optional list of names (uses filenames if None)
        """
        if names and len(names) != len(filepaths):
            raise ValueError("Length of names must match filepaths")
        
        for i, filepath in enumerate(filepaths):
            name = names[i] if names else None
            self.load_result(filepath, name)
    
    def print_summary(
        self,
        include_sections: Optional[List[str]] = None,
        exclude_sections: Optional[List[str]] = None,
        metric_filter: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        reverse_sort: bool = False
    ):
        """
        Print formatted summary of all loaded results.
        
        Args:
            include_sections: Only show these sections (None = all)
            exclude_sections: Hide these sections
            metric_filter: Only show these metrics
            sort_by: Sort results by this metric
            reverse_sort: Sort in descending order
        """
        if not self.results:
            print("No results loaded.")
            return
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(f"\nComparing {len(self.results)} result(s)")
        
        # Determine sections to show
        all_sections = set()
        for result in self.results:
            all_sections.update(result.keys())
        
        sections_to_show = all_sections
        if include_sections:
            sections_to_show = set(include_sections) & all_sections
        if exclude_sections:
            sections_to_show = sections_to_show - set(exclude_sections)
        
        # Sort results if requested
        display_order = list(range(len(self.results)))
        if sort_by:
            display_order = self._get_sorted_order(sort_by, reverse_sort)
        
        # Print each section
        for section in sorted(sections_to_show):
            self._print_section(section, display_order, metric_filter)
    
    def _get_sorted_order(self, metric: str, reverse: bool) -> List[int]:
        """Get indices sorted by metric value."""
        values = []
        for i, result in enumerate(self.results):
            value = self._find_metric_value(result, metric)
            values.append((i, value if value is not None else float('-inf')))
        
        sorted_values = sorted(values, key=lambda x: x[1], reverse=reverse)
        return [i for i, _ in sorted_values]
    
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
    
    def _print_section(
        self,
        section: str,
        display_order: List[int],
        metric_filter: Optional[List[str]]
    ):
        """Print a single section."""
        # Check if section exists in any result
        section_data = []
        for i in display_order:
            if section in self.results[i]:
                section_data.append(self.results[i][section])
            else:
                section_data.append(None)
        
        if all(d is None for d in section_data):
            return
        
        print(f"\n{'─'*80}")
        print(f"{section.upper().replace('_', ' ')}")
        print(f"{'─'*80}")
        
        # If section contains nested dict (like efficiency, performance, retrieval)
        if any(isinstance(d, dict) for d in section_data if d is not None):
            self._print_nested_section(section_data, display_order, metric_filter)
        else:
            # Simple value
            for i in display_order:
                value = section_data[display_order.index(i)]
                print(f"  {self.result_names[i]:.<40} {self._format_value(value)}")
    
    def _print_nested_section(
        self,
        section_data: List[Optional[Dict]],
        display_order: List[int],
        metric_filter: Optional[List[str]]
    ):
        """Print nested dictionary section."""
        # Collect all metrics across all results
        all_metrics = set()
        for data in section_data:
            if isinstance(data, dict):
                all_metrics.update(data.keys())
        
        if metric_filter:
            all_metrics = all_metrics & set(metric_filter)
        
        # Print each metric
        for metric in sorted(all_metrics):
            # Check if this is a nested dict itself
            is_nested = False
            for data in section_data:
                if isinstance(data, dict) and metric in data:
                    if isinstance(data[metric], dict):
                        is_nested = True
                        break
            
            if is_nested:
                # Nested dict (like precision_at_k)
                print(f"\n  {metric}:")
                self._print_nested_dict_metric(metric, section_data, display_order)
            else:
                # Simple metric
                print(f"\n  {metric}:")
                for i in display_order:
                    data = section_data[display_order.index(i)]
                    value = data.get(metric) if isinstance(data, dict) else None
                    formatted = self._format_value(value)
                    print(f"    {self.result_names[i]:.<38} {formatted}")
    
    def _print_nested_dict_metric(
        self,
        metric: str,
        section_data: List[Optional[Dict]],
        display_order: List[int]
    ):
        """Print nested dictionary metric (e.g., precision_at_k with k values)."""
        # Collect all sub-keys
        all_subkeys = set()
        for data in section_data:
            if isinstance(data, dict) and metric in data:
                if isinstance(data[metric], dict):
                    all_subkeys.update(data[metric].keys())
        
        for subkey in sorted(all_subkeys):
            print(f"    {subkey}:")
            for i in display_order:
                data = section_data[display_order.index(i)]
                if isinstance(data, dict) and metric in data:
                    nested = data[metric]
                    if isinstance(nested, dict):
                        value = nested.get(subkey)
                        formatted = self._format_value(value)
                        print(f"      {self.result_names[i]:.<36} {formatted}")
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "N/A"
        elif isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, dict):
            return f"{len(value)} items"
        elif isinstance(value, list):
            return f"{len(value)} items"
        else:
            return str(value)
    
    def get_comparison_table(
        self,
        metrics: List[str],
        section: Optional[str] = None
    ) -> str:
        """
        Generate a comparison table for specific metrics.
        
        Args:
            metrics: List of metric names
            section: Optional section to look in (None = search all)
            
        Returns:
            Formatted table string
        """
        if not self.results:
            return "No results loaded."
        
        # Header
        col_width = max(len(name) for name in self.result_names) + 2
        header = f"{'Metric':<30}"
        for name in self.result_names:
            header += f"{name:>{col_width}}"
        
        lines = [header, "─" * len(header)]
        
        # Data rows
        for metric in metrics:
            row = f"{metric:<30}"
            for result in self.results:
                value = self._find_metric_value(result, metric)
                if value is not None:
                    row += f"{value:>{col_width}.4f}"
                else:
                    row += f"{'N/A':>{col_width}}"
            lines.append(row)
        
        return "\n".join(lines)
    
    def print_best_worst(
        self,
        metric: str,
        higher_is_better: bool = True,
        top_n: int = 3
    ):
        """
        Print best and worst performing models for a metric.
        
        Args:
            metric: Metric name
            higher_is_better: Whether higher values are better
            top_n: Number of top/bottom models to show
        """
        values = []
        for i, result in enumerate(self.results):
            value = self._find_metric_value(result, metric)
            if value is not None:
                values.append((self.result_names[i], value))
        
        if not values:
            print(f"Metric '{metric}' not found in any results.")
            return
        
        sorted_values = sorted(values, key=lambda x: x[1], reverse=higher_is_better)
        
        print(f"\n{'='*60}")
        print(f"RANKING: {metric}")
        print(f"{'='*60}")
        
        print(f"\n{'Best' if higher_is_better else 'Lowest'} {top_n}:")
        for i, (name, value) in enumerate(sorted_values[:top_n], 1):
            print(f"  {i}. {name:.<40} {value:.4f}")
        
        if len(sorted_values) > top_n:
            print(f"\n{'Worst' if higher_is_better else 'Highest'} {top_n}:")
            for i, (name, value) in enumerate(reversed(sorted_values[-top_n:]), 1):
                print(f"  {i}. {name:.<40} {value:.4f}")
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export loaded results as a dictionary."""
        return {
            'results': self.results,
            'names': self.result_names
        }


def main():
    """CLI interface for summary tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Summarize evaluation results')
    parser.add_argument('files', nargs='+', help='JSON result files to summarize')
    parser.add_argument('--names', nargs='+', help='Custom names for results')
    parser.add_argument('--include', nargs='+', help='Only show these sections')
    parser.add_argument('--exclude', nargs='+', help='Hide these sections')
    parser.add_argument('--metrics', nargs='+', help='Only show these metrics')
    parser.add_argument('--sort-by', help='Sort results by metric')
    parser.add_argument('--reverse', action='store_true', help='Reverse sort order')
    parser.add_argument('--table', nargs='+', help='Generate comparison table for metrics')
    parser.add_argument('--best-worst', help='Show best/worst for metric')
    parser.add_argument('--higher-better', action='store_true', default=True,
                       help='For best-worst, higher is better')
    
    args = parser.parse_args()
    
    summary = ResultsSummary()
    summary.load_results(args.files, args.names)
    
    if args.table:
        print(summary.get_comparison_table(args.table))
    elif args.best_worst:
        summary.print_best_worst(args.best_worst, args.higher_better)
    else:
        summary.print_summary(
            include_sections=args.include,
            exclude_sections=args.exclude,
            metric_filter=args.metrics,
            sort_by=args.sort_by,
            reverse_sort=args.reverse
        )


if __name__ == "__main__":
    main()
