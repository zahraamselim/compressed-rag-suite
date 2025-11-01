"""Wrapper for lm-evaluation-harness with comprehensive task support."""

import logging
from typing import List, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

# Task registry with metadata
TASK_REGISTRY = {
    # Language Understanding
    'lambada': {
        'description': 'Language modeling task',
        'metric': 'acc',
        'default_fewshot': 0
    },
    'storycloze': {
        'description': 'Story completion',
        'metric': 'acc',
        'default_fewshot': 0
    },
    
    # Reasoning & Common Sense
    'arc_easy': {
        'description': 'AI2 Reasoning Challenge (Easy)',
        'metric': 'acc_norm',
        'default_fewshot': 0
    },
    'arc_challenge': {
        'description': 'AI2 Reasoning Challenge (Challenge)',
        'metric': 'acc_norm',
        'default_fewshot': 25
    },
    'commonsense_qa': {
        'description': 'CommonsenseQA',
        'metric': 'acc',
        'default_fewshot': 0
    },
    'hellaswag': {
        'description': 'HellaSwag commonsense reasoning',
        'metric': 'acc_norm',
        'default_fewshot': 10
    },
    'openbookqa': {
        'description': 'OpenBookQA',
        'metric': 'acc_norm',
        'default_fewshot': 0
    },
    'piqa': {
        'description': 'Physical Interaction QA',
        'metric': 'acc_norm',
        'default_fewshot': 0
    },
    'siqa': {
        'description': 'Social Interaction QA',
        'metric': 'acc',
        'default_fewshot': 0
    },
    'winogrande': {
        'description': 'Winograd Schema Challenge',
        'metric': 'acc',
        'default_fewshot': 5
    },
    
    # Knowledge & QA
    'nq_open': {
        'description': 'Natural Questions (Open)',
        'metric': 'exact_match',
        'default_fewshot': 0
    },
    'triviaqa': {
        'description': 'TriviaQA',
        'metric': 'exact_match',
        'default_fewshot': 5
    },
    
    # Comprehensive Benchmarks
    'mmlu': {
        'description': 'Massive Multitask Language Understanding',
        'metric': 'acc',
        'default_fewshot': 5
    },
    'bbh': {
        'description': 'BIG-Bench Hard',
        'metric': 'acc',
        'default_fewshot': 3
    },
    'agieval': {
        'description': 'AGIEval',
        'metric': 'acc',
        'default_fewshot': 0
    },
    
    # GLUE & SuperGLUE
    'glue': {
        'description': 'GLUE benchmark (all tasks)',
        'metric': 'acc',
        'default_fewshot': 0
    },
    'super_glue': {
        'description': 'SuperGLUE benchmark (all tasks)',
        'metric': 'acc',
        'default_fewshot': 0
    },
    'boolq': {
        'description': 'BoolQ (boolean questions)',
        'metric': 'acc',
        'default_fewshot': 0
    },
    
    # Math & Code
    'gsm8k': {
        'description': 'Grade School Math 8K',
        'metric': 'exact_match',
        'default_fewshot': 5
    },
    'hendrycks_math': {
        'description': 'MATH dataset',
        'metric': 'exact_match',
        'default_fewshot': 4
    },
    'math_algebra': {
        'description': 'MATH Algebra subset',
        'metric': 'exact_match',
        'default_fewshot': 4
    },
    'humaneval': {
        'description': 'HumanEval code generation',
        'metric': 'pass@1',
        'default_fewshot': 0
    },
    'mbpp': {
        'description': 'Mostly Basic Python Problems',
        'metric': 'pass@1',
        'default_fewshot': 3
    }
}


def parse_task_config(task_config: Union[bool, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Parse task configuration from config file.
    
    Args:
        task_config: Either boolean (enabled/disabled) or dict with settings
        
    Returns:
        Task configuration dict or None if disabled
    """
    if isinstance(task_config, bool):
        return {'enabled': task_config} if task_config else None
    elif isinstance(task_config, dict):
        if not task_config.get('enabled', True):
            return None
        return task_config
    return None


def get_metric_from_results(task_results: Dict[str, Any], task_name: str) -> Optional[float]:
    """
    Extract the appropriate metric from task results.
    
    Args:
        task_results: Results dictionary for a task
        task_name: Name of the task
        
    Returns:
        Extracted metric value or None
    """
    # Get expected metric for this task
    task_info = TASK_REGISTRY.get(task_name, {})
    preferred_metric = task_info.get('metric', 'acc')
    
    # Priority order for metric names (with variations)
    metric_variations = [
        preferred_metric,
        f"{preferred_metric},none",
        f"{preferred_metric}_norm",
        f"{preferred_metric}_norm,none",
        "acc_norm",
        "acc_norm,none",
        "acc",
        "acc,none",
        "exact_match",
        "exact_match,none",
        "pass@1",
        "f1",
        "em"
    ]
    
    for metric_name in metric_variations:
        if metric_name in task_results:
            value = task_results[metric_name]
            
            # Handle both direct values and nested dicts
            if isinstance(value, dict):
                # Try different keys in order of preference
                for key in ['mean', 'value', 'score']:
                    if key in value:
                        return float(value[key])
            elif isinstance(value, (int, float)):
                return float(value)
    
    # Fallback: find any numeric value
    for key, value in task_results.items():
        if isinstance(value, (int, float)):
            logger.debug(f"Using fallback metric '{key}' for task {task_name}")
            return float(value)
        elif isinstance(value, dict):
            for subkey in ['mean', 'value', 'score']:
                if subkey in value and isinstance(value[subkey], (int, float)):
                    logger.debug(f"Using fallback metric '{key}.{subkey}' for task {task_name}")
                    return float(value[subkey])
    
    return None


def run_lm_eval_harness(
    model_interface,
    tasks: Union[List[str], Dict[str, Any]],
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Run tasks using lm-evaluation-harness.
    
    Args:
        model_interface: ModelInterface instance
        tasks: Either list of task names or dict with task configs
        num_fewshot: Default number of few-shot examples
        limit: Limit number of samples per task
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary mapping task names to scores
    """
    try:
        # Try new API first (v0.4.0+)
        try:
            from lm_eval import simple_evaluate
            use_new_api = True
        except ImportError:
            # Fall back to old API
            from lm_eval import evaluator
            use_new_api = False
        
        # Parse tasks configuration
        task_list = []
        task_configs = {}
        
        if isinstance(tasks, dict):
            # Dict format: {task_name: {enabled: true, num_fewshot: 5, ...}}
            for task_name, task_config in tasks.items():
                parsed_config = parse_task_config(task_config)
                if parsed_config is not None:
                    task_list.append(task_name)
                    task_configs[task_name] = parsed_config
        elif isinstance(tasks, list):
            # List format: ["task1", "task2", ...]
            task_list = tasks
            task_configs = {task: {} for task in tasks}
        else:
            logger.error(f"Invalid tasks format: {type(tasks)}")
            return {}
        
        if not task_list:
            logger.warning("No tasks enabled for evaluation")
            return {}
        
        logger.info(f"Running lm-eval-harness for {len(task_list)} tasks: {task_list}")
        logger.info(f"API version: {'new (v0.4.0+)' if use_new_api else 'old (pre-v0.4.0)'}")
        
        # Get lm-eval compatible model from interface
        lm_eval_model = model_interface.get_lm_eval_model()
        
        if lm_eval_model is None:
            logger.error("Model interface does not support lm-eval")
            return {}
        
        # Run evaluation with appropriate API
        all_metrics = {}
        
        # Process tasks individually or in groups
        for task_name in task_list:
            try:
                # Get task-specific configuration
                task_cfg = task_configs.get(task_name, {})
                task_fewshot = task_cfg.get('num_fewshot', num_fewshot)
                task_limit = task_cfg.get('limit', limit)
                task_batch = task_cfg.get('batch_size', batch_size)
                
                # Use default fewshot from registry if not specified
                if task_fewshot == 0 and task_name in TASK_REGISTRY:
                    task_fewshot = TASK_REGISTRY[task_name].get('default_fewshot', 0)
                
                logger.info(f"Evaluating {task_name} (fewshot={task_fewshot})")
                
                # Run evaluation
                if use_new_api:
                    results = simple_evaluate(
                        model=lm_eval_model,
                        tasks=[task_name],
                        num_fewshot=task_fewshot,
                        limit=task_limit,
                        batch_size=task_batch
                    )
                else:
                    # Old API
                    results = evaluator.simple_evaluate(
                        model=lm_eval_model,
                        tasks=[task_name],
                        num_fewshot=task_fewshot,
                        limit=task_limit,
                        batch_size=task_batch
                    )
                
                # Extract metrics for this task
                if task_name in results.get("results", {}):
                    task_results = results["results"][task_name]
                    score = get_metric_from_results(task_results, task_name)
                    
                    if score is not None:
                        all_metrics[task_name] = score
                        logger.info(f"✓ {task_name}: {score:.4f}")
                    else:
                        logger.warning(f"✗ {task_name}: No valid metric found")
                        logger.debug(f"Available results: {list(task_results.keys())}")
                
            except Exception as e:
                logger.error(f"Error evaluating task {task_name}: {e}")
                continue
        
        if not all_metrics:
            logger.warning("No metrics extracted from lm-eval results")
        else:
            logger.info(f"Successfully evaluated {len(all_metrics)}/{len(task_list)} tasks")
        
        return all_metrics
        
    except ImportError as e:
        logger.error(f"lm-eval not installed or incompatible version: {e}")
        logger.error("Install with: pip install lm-eval")
        return {}
    except Exception as e:
        logger.error(f"Error running lm-eval-harness: {e}", exc_info=True)
        return {}


def list_available_tasks() -> Dict[str, str]:
    """
    Get list of all supported tasks with descriptions.
    
    Returns:
        Dictionary mapping task names to descriptions
    """
    return {task: info['description'] for task, info in TASK_REGISTRY.items()}


def get_task_info(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Task information dict or None if not found
    """
    return TASK_REGISTRY.get(task_name)
