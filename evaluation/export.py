"""Export tool for evaluation results in various formats."""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultsExporter:
    """Export evaluation results to various formats."""
    
    def __init__(self):
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
    
    def _find_metric_value(self, result: Dict, metric: str) -> Any:
        """Find a metric value in nested dictionary."""
        # Direct key
        if metric in result:
            return result[metric]
        
        # Search in nested dicts
        for key, value in result.items():
            if isinstance(value, dict) and metric in value:
                return value[metric]
        
        return None
    
    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export_to_csv(
        self,
        output_file: str,
        metrics: Optional[List[str]] = None,
        flatten: bool = True
    ):
        """
        Export results to CSV.
        
        Args:
            output_file: Output CSV file path
            metrics: Specific metrics to export (None = all)
            flatten: Flatten nested dictionaries
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        if flatten:
            flattened_results = [self._flatten_dict(r) for r in self.results]
        else:
            flattened_results = self.results
        
        # Collect all keys
        all_keys = set()
        for result in flattened_results:
            all_keys.update(result.keys())
        
        if metrics:
            all_keys = all_keys & set(metrics)
        
        all_keys = sorted(all_keys)
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Model'] + all_keys)
            
            # Data rows
            for name, result in zip(self.result_names, flattened_results):
                row = [name]
                for key in all_keys:
                    value = result.get(key, '')
                    # Format value
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    elif isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    row.append(value)
                writer.writerow(row)
        
        logger.info(f"Exported to CSV: {output_path}")
    
    def export_to_markdown(
        self,
        output_file: str,
        metrics: Optional[List[str]] = None,
        title: str = "Evaluation Results"
    ):
        """
        Export results to Markdown table.
        
        Args:
            output_file: Output Markdown file path
            metrics: Specific metrics to export (None = auto-select)
            title: Table title
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-select important metrics if not provided
        if metrics is None:
            metrics = self._auto_select_metrics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Title
            f.write(f"# {title}\n\n")
            f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Table header
            f.write("| Model | " + " | ".join(metrics) + " |\n")
            f.write("|" + "---|" * (len(metrics) + 1) + "\n")
            
            # Table rows
            for name, result in zip(self.result_names, self.results):
                row = f"| {name} |"
                for metric in metrics:
                    value = self._find_metric_value(result, metric)
                    if value is None:
                        formatted = " N/A"
                    elif isinstance(value, float):
                        formatted = f" {value:.4f}"
                    else:
                        formatted = f" {value}"
                    row += formatted + " |"
                f.write(row + "\n")
            
            f.write("\n")
        
        logger.info(f"Exported to Markdown: {output_path}")
    
    def export_to_latex(
        self,
        output_file: str,
        metrics: Optional[List[str]] = None,
        caption: str = "Evaluation Results",
        label: str = "tab:results"
    ):
        """
        Export results to LaTeX table.
        
        Args:
            output_file: Output .tex file path
            metrics: Specific metrics to export
            caption: Table caption
            label: Table label for references
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if metrics is None:
            metrics = self._auto_select_metrics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Table begin
            col_format = 'l' + 'r' * len(metrics)
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")
            f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
            f.write("\\toprule\n")
            
            # Header
            header = "Model & " + " & ".join(self._latex_escape(m) for m in metrics) + " \\\\\n"
            f.write(header)
            f.write("\\midrule\n")
            
            # Data rows
            for name, result in zip(self.result_names, self.results):
                row = self._latex_escape(name)
                for metric in metrics:
                    value = self._find_metric_value(result, metric)
                    if value is None:
                        formatted = " & ---"
                    elif isinstance(value, float):
                        formatted = f" & {value:.4f}"
                    else:
                        formatted = f" & {self._latex_escape(str(value))}"
                    row += formatted
                row += " \\\\\n"
                f.write(row)
            
            # Table end
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"Exported to LaTeX: {output_path}")
    
    def export_to_html(
        self,
        output_file: str,
        metrics: Optional[List[str]] = None,
        title: str = "Evaluation Results",
        include_style: bool = True
    ):
        """
        Export results to HTML table.
        
        Args:
            output_file: Output HTML file path
            metrics: Specific metrics to export
            title: Page title
            include_style: Include CSS styling
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if metrics is None:
            metrics = self._auto_select_metrics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # HTML header
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write(f"<title>{title}</title>\n")
            f.write('<meta charset="UTF-8">\n')
            
            if include_style:
                f.write("<style>\n")
                f.write("""
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                    th { background-color: #4CAF50; color: white; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    tr:hover { background-color: #ddd; }
                    .best { background-color: #d4edda !important; font-weight: bold; }
                    .timestamp { color: #666; font-size: 0.9em; }
                """)
                f.write("</style>\n")
            
            f.write("</head>\n<body>\n")
            
            # Content
            f.write(f"<h1>{title}</h1>\n")
            f.write(f'<p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
            
            # Table
            f.write("<table>\n<thead>\n<tr>\n")
            f.write("<th>Model</th>\n")
            for metric in metrics:
                f.write(f"<th>{self._html_escape(metric)}</th>\n")
            f.write("</tr>\n</thead>\n<tbody>\n")
            
            # Find best values for highlighting
            best_values = {}
            for metric in metrics:
                values = []
                for result in self.results:
                    value = self._find_metric_value(result, metric)
                    if isinstance(value, (int, float)):
                        values.append(value)
                if values:
                    best_values[metric] = max(values)  # Assume higher is better
            
            # Data rows
            for name, result in zip(self.result_names, self.results):
                f.write("<tr>\n")
                f.write(f"<td><strong>{self._html_escape(name)}</strong></td>\n")
                
                for metric in metrics:
                    value = self._find_metric_value(result, metric)
                    is_best = (isinstance(value, (int, float)) and 
                              metric in best_values and 
                              value == best_values[metric])
                    
                    cell_class = ' class="best"' if is_best else ''
                    
                    if value is None:
                        formatted = "N/A"
                    elif isinstance(value, float):
                        formatted = f"{value:.4f}"
                    else:
                        formatted = str(value)
                    
                    f.write(f"<td{cell_class}>{self._html_escape(formatted)}</td>\n")
                
                f.write("</tr>\n")
            
            f.write("</tbody>\n</table>\n")
            f.write("</body>\n</html>\n")
        
        logger.info(f"Exported to HTML: {output_path}")
    
    def _auto_select_metrics(self) -> List[str]:
        """Auto-select important metrics for export."""
        important = [
            # Efficiency
            'latency_ms_per_token',
            'throughput_tokens_per_sec',
            'peak_memory_mb',
            'energy_per_token_mj',
            # Performance
            'perplexity',
            'average_accuracy',
            # Retrieval
            'precision_at_3',
            'exact_match',
            'f1_score',
            'faithfulness'
        ]
        
        # Only include metrics that exist
        available = []
        for metric in important:
            for result in self.results:
                if self._find_metric_value(result, metric) is not None:
                    available.append(metric)
                    break
        
        return available
    
    def _latex_escape(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '\\': '\\textbackslash{}',
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _html_escape(self, text: str) -> str:
        """Escape special HTML characters."""
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def export_all_formats(
        self,
        output_dir: str,
        base_name: str = "results",
        metrics: Optional[List[str]] = None
    ):
        """
        Export to all formats at once.
        
        Args:
            output_dir: Output directory
            base_name: Base filename (without extension)
            metrics: Specific metrics to export
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting to all formats in {output_dir}...")
        
        self.export_to_csv(
            str(output_path / f"{base_name}.csv"),
            metrics
        )
        
        self.export_to_markdown(
            str(output_path / f"{base_name}.md"),
            metrics
        )
        
        self.export_to_latex(
            str(output_path / f"{base_name}.tex"),
            metrics
        )
        
        self.export_to_html(
            str(output_path / f"{base_name}.html"),
            metrics
        )
        
        logger.info("Export complete!")


def main():
    """CLI interface for exporter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export evaluation results')
    parser.add_argument('files', nargs='+', help='JSON result files')
    parser.add_argument('--names', nargs='+', help='Custom names for results')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--format', choices=['csv', 'md', 'latex', 'html', 'all'],
                       default='csv', help='Output format')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to export')
    parser.add_argument('--title', default='Evaluation Results',
                       help='Title for document')
    
    args = parser.parse_args()
    
    exporter = ResultsExporter()
    exporter.load_results(args.files, args.names)
    
    if args.format == 'csv':
        exporter.export_to_csv(args.output, args.metrics)
    elif args.format == 'md':
        exporter.export_to_markdown(args.output, args.metrics, args.title)
    elif args.format == 'latex':
        exporter.export_to_latex(args.output, args.metrics, args.title)
    elif args.format == 'html':
        exporter.export_to_html(args.output, args.metrics, args.title)
    elif args.format == 'all':
        output_path = Path(args.output)
        exporter.export_all_formats(
            str(output_path.parent),
            output_path.stem,
            args.metrics
        )


if __name__ == "__main__":
    main()
