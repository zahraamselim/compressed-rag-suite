"""Utility functions for managing LM-Eval tasks."""

import json
from typing import Dict, List, Optional
from pathlib import Path

from evaluation.performance.lm_eval_wrapper import TASK_REGISTRY, list_available_tasks, get_task_info


def print_available_tasks():
    """Print all available tasks with descriptions."""
    print("\n" + "="*80)
    print("AVAILABLE LM-EVAL TASKS")
    print("="*80)
    
    categories = {
        'Reasoning & Common Sense': [
            'arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 
            'siqa', 'winogrande', 'commonsense_qa', 'openbookqa'
        ],
        'Language Understanding': ['lambada', 'storycloze'],
        'Knowledge & QA': ['nq_open', 'triviaqa'],
        'Comprehensive Benchmarks': ['mmlu', 'bbh', 'agieval'],
        'GLUE/SuperGLUE': ['glue', 'super_glue', 'boolq'],
        'Math': ['gsm8k', 'hendrycks_math', 'math_algebra'],
        'Code Generation': ['humaneval', 'mbpp']
    }
    
    for category, task_list in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for task_name in task_list:
            info = get_task_info(task_name)
            if info:
                print(f"  • {task_name:20} - {info['description']}")
                print(f"    {'':20}   Metric: {info['metric']}, Default few-shot: {info['default_fewshot']}")
    
    print("\n" + "="*80 + "\n")


def generate_config_template(
    output_file: str = "lm_eval_config_template.json",
    enabled_by_default: bool = False
) -> str:
    """
    Generate a template configuration file for all tasks.
    
    Args:
        output_file: Path to save the template
        enabled_by_default: Whether tasks should be enabled by default
        
    Returns:
        Path to generated file
    """
    config = {"lm_eval_tasks": {}}
    
    for task_name, info in TASK_REGISTRY.items():
        config["lm_eval_tasks"][task_name] = {
            "enabled": enabled_by_default,
            "num_fewshot": info['default_fewshot'],
            "limit": None,
            "batch_size": 1
        }
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Template saved to: {output_path}")
    return str(output_path)


def generate_quick_config(
    tasks: List[str],
    output_file: str = "lm_eval_quick_config.json"
) -> str:
    """
    Generate a quick config for specified tasks with recommended settings.
    
    Args:
        tasks: List of task names to enable
        output_file: Path to save the config
        
    Returns:
        Path to generated file
    """
    config = {"lm_eval_tasks": {}}
    
    for task_name in tasks:
        info = get_task_info(task_name)
        if info:
            config["lm_eval_tasks"][task_name] = {
                "enabled": True,
                "num_fewshot": info['default_fewshot']
            }
        else:
            print(f"⚠ Warning: Unknown task '{task_name}'")
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Quick config saved to: {output_path}")
    return str(output_path)


def get_preset_configs() -> Dict[str, List[str]]:
    """
    Get predefined task presets for common evaluation scenarios.
    
    Returns:
        Dictionary mapping preset names to task lists
    """
    return {
        'baseline': [
            'hellaswag', 'piqa', 'arc_easy'
        ],
        'standard': [
            'hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 
            'winogrande', 'lambada'
        ],
        'comprehensive': [
            'hellaswag', 'piqa', 'arc_easy', 'arc_challenge',
            'winogrande', 'mmlu', 'bbh', 'lambada', 'storycloze'
        ],
        'reasoning': [
            'hellaswag', 'arc_challenge', 'winogrande', 
            'bbh', 'commonsense_qa'
        ],
        'knowledge': [
            'mmlu', 'triviaqa', 'nq_open', 'agieval'
        ],
        'math_code': [
            'gsm8k', 'hendrycks_math', 'humaneval', 'mbpp'
        ],
        'nlp_benchmarks': [
            'glue', 'super_glue', 'boolq'
        ],
        'quick_test': [
            'hellaswag', 'piqa', 'arc_easy'
        ]
    }


def generate_preset_config(
    preset_name: str,
    output_file: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """
    Generate config from a preset.
    
    Args:
        preset_name: Name of the preset ('baseline', 'standard', etc.)
        output_file: Path to save config (auto-generated if None)
        limit: Optional limit for all tasks
        
    Returns:
        Path to generated file
    """
    presets = get_preset_configs()
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    tasks = presets[preset_name]
    config = {"lm_eval_tasks": {}}
    
    for task_name in tasks:
        info = get_task_info(task_name)
        if info:
            task_config = {
                "enabled": True,
                "num_fewshot": info['default_fewshot']
            }
            if limit is not None:
                task_config["limit"] = limit
            
            config["lm_eval_tasks"][task_name] = task_config
    
    if output_file is None:
        output_file = f"lm_eval_{preset_name}_config.json"
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Preset '{preset_name}' config saved to: {output_path}")
    print(f"  Tasks: {', '.join(tasks)}")
    return str(output_path)


def validate_config(config_dict: Dict) -> List[str]:
    """
    Validate a task configuration.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []
    
    lm_eval_tasks = config_dict.get('lm_eval_tasks', {})
    
    if not lm_eval_tasks:
        issues.append("No tasks configured")
        return issues
    
    enabled_count = 0
    
    for task_name, task_config in lm_eval_tasks.items():
        # Check if task exists
        if task_name not in TASK_REGISTRY:
            issues.append(f"Unknown task: '{task_name}'")
            continue
        
        # Parse config
        if isinstance(task_config, bool):
            if task_config:
                enabled_count += 1
        elif isinstance(task_config, dict):
            if task_config.get('enabled', True):
                enabled_count += 1
            
            # Validate config fields
            num_fewshot = task_config.get('num_fewshot')
            if num_fewshot is not None and (not isinstance(num_fewshot, int) or num_fewshot < 0):
                issues.append(f"{task_name}: num_fewshot must be non-negative integer")
            
            limit = task_config.get('limit')
            if limit is not None and (not isinstance(limit, int) or limit <= 0):
                issues.append(f"{task_name}: limit must be positive integer or null")
            
            batch_size = task_config.get('batch_size')
            if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
                issues.append(f"{task_name}: batch_size must be positive integer")
    
    if enabled_count == 0:
        issues.append("No tasks enabled")
    
    return issues


def print_presets():
    """Print all available presets."""
    print("\n" + "="*80)
    print("AVAILABLE PRESETS")
    print("="*80)
    
    presets = get_preset_configs()
    
    for preset_name, tasks in presets.items():
        print(f"\n{preset_name}:")
        print(f"  Tasks: {', '.join(tasks)}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python lm_eval_utils.py list                    - List all tasks")
        print("  python lm_eval_utils.py presets                 - List presets")
        print("  python lm_eval_utils.py template [--enabled]    - Generate template")
        print("  python lm_eval_utils.py preset <name> [--limit N] - Generate preset config")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "list":
        print_available_tasks()
    elif command == "presets":
        print_presets()
    elif command == "template":
        enabled = "--enabled" in sys.argv
        generate_config_template(enabled_by_default=enabled)
    elif command == "preset":
        if len(sys.argv) < 3:
            print("Error: Please specify preset name")
            print_presets()
            sys.exit(1)
        
        preset_name = sys.argv[2]
        limit = None
        if "--limit" in sys.argv:
            limit_idx = sys.argv.index("--limit")
            if limit_idx + 1 < len(sys.argv):
                limit = int(sys.argv[limit_idx + 1])
        
        generate_preset_config(preset_name, limit=limit)
    else:
        print(f"Unknown command: {command}")
