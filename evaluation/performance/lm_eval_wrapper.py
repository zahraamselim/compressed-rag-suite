"""Wrapper for lm-evaluation-harness with comprehensive task support."""

import logging
from typing import List, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

# Comprehensive task registry with Mistral paper defaults
TASK_REGISTRY = {
    # Commonsense Reasoning (0-shot)
    'hellaswag': {
        'description': 'HellaSwag - sentence completion',
        'metric': 'acc_norm',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    'winogrande': {
        'description': 'Winogrande - pronoun resolution',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    'piqa': {
        'description': 'PIQA - physical interactions',
        'metric': 'acc_norm',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    'siqa': {
        'description': 'SIQA - social interactions',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    'openbookqa': {
        'description': 'OpenBookQA - elementary science',
        'metric': 'acc_norm',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    'arc_easy': {
        'description': 'ARC-Easy - science questions',
        'metric': 'acc_norm',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    'arc_challenge': {
        'description': 'ARC-Challenge - hard science',
        'metric': 'acc_norm',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    'commonsense_qa': {
        'description': 'CommonsenseQA - general knowledge',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'commonsense'
    },
    
    # World Knowledge (5-shot)
    'nq_open': {
        'description': 'Natural Questions - open domain',
        'metric': 'exact_match',
        'default_fewshot': 5,
        'category': 'knowledge'
    },
    'triviaqa': {
        'description': 'TriviaQA - trivia questions',
        'metric': 'exact_match',
        'default_fewshot': 5,
        'category': 'knowledge'
    },
    
    # Reading Comprehension (0-shot)
    'boolq': {
        'description': 'BoolQ - boolean questions',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'reading'
    },
    'quac': {
        'description': 'QuAC - conversational QA',
        'metric': 'f1',
        'default_fewshot': 0,
        'category': 'reading'
    },
    
    # Math
    'gsm8k': {
        'description': 'GSM8K - grade school math (maj@8)',
        'metric': 'exact_match',
        'default_fewshot': 8,
        'category': 'math'
    },
    'hendrycks_math': {
        'description': 'MATH - competition math (maj@4)',
        'metric': 'exact_match',
        'default_fewshot': 4,
        'category': 'math'
    },
    'math_algebra': {
        'description': 'MATH Algebra subset',
        'metric': 'exact_match',
        'default_fewshot': 4,
        'category': 'math'
    },
    
    # Code (0-shot and 3-shot)
    'humaneval': {
        'description': 'HumanEval - Python code (pass@1)',
        'metric': 'pass@1',
        'default_fewshot': 0,
        'category': 'code'
    },
    'mbpp': {
        'description': 'MBPP - Python problems',
        'metric': 'pass@1',
        'default_fewshot': 3,
        'category': 'code'
    },
    
    # Aggregate Benchmarks
    'mmlu': {
        'description': 'MMLU - multitask understanding (5-shot)',
        'metric': 'acc',
        'default_fewshot': 5,
        'category': 'aggregate'
    },
    'bbh': {
        'description': 'BBH - BIG-Bench Hard (3-shot)',
        'metric': 'acc',
        'default_fewshot': 3,
        'category': 'aggregate'
    },
    'agieval': {
        'description': 'AGI Eval - English MCQ (3-5 shot)',
        'metric': 'acc',
        'default_fewshot': 3,
        'category': 'aggregate'
    },
    
    # Language Understanding
    'lambada': {
        'description': 'LAMBADA - word prediction',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'language'
    },
    'storycloze': {
        'description': 'StoryCloze - story completion',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'language'
    },
    
    # NLP Benchmarks
    'glue': {
        'description': 'GLUE benchmark suite',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'nlp'
    },
    'super_glue': {
        'description': 'SuperGLUE benchmark suite',
        'metric': 'acc',
        'default_fewshot': 0,
        'category': 'nlp'
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
    Handles various lm-eval output formats.
    
    Args:
        task_results: Results dictionary for a task
        task_name: Name of the task
        
    Returns:
        Extracted metric value or None
    """
    # Get expected metric for this task
    task_info = TASK_REGISTRY.get(task_name, {})
    preferred_metric = task_info.get('metric', 'acc')
    
    # Priority order for metric names (with common variations)
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
    
    logger.warning(f"No valid metric found for {task_name}. Available keys: {list(task_results.keys())}")
    return None


def run_lm_eval_harness(
    model_interface,
    tasks: Union[List[str], Dict[str, Any]],
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Run tasks using lm-evaluation-harness.
    
    Args:
        model_interface: ModelInterface instance
        tasks: Either list of task names or dict with task configs
        num_fewshot: Default number of few-shot examples (None = use task defaults)
        limit: Limit number of samples per task
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary mapping task names to scores (0.0 to 1.0)
    """
    try:
        # Try new API first (v0.4.0+)
        try:
            from lm_eval import simple_evaluate
            use_new_api = True
            logger.debug("Using lm-eval API v0.4.0+")
        except ImportError:
            # Fall back to old API
            try:
                from lm_eval import evaluator
                use_new_api = False
                logger.debug("Using lm-eval API pre-v0.4.0")
            except ImportError:
                logger.error("lm-eval not installed. Install with: pip install lm-eval")
                return {}
        
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
        
        logger.info(f"Running lm-eval for {len(task_list)} tasks")
        
        # Get lm-eval compatible model from interface
        lm_eval_model = model_interface.get_lm_eval_model()
        
        if lm_eval_model is None:
            logger.error("Model interface does not support lm-eval")
            logger.error("Implement get_lm_eval_model() in your ModelInterface")
            return {}
        
        # Run evaluation - process tasks individually for better error handling
        all_metrics = {}
        
        for task_name in task_list:
            try:
                # Get task-specific configuration
                task_cfg = task_configs.get(task_name, {})
                
                # Determine few-shot setting
                if 'num_fewshot' in task_cfg:
                    task_fewshot = task_cfg['num_fewshot']
                elif num_fewshot is not None:
                    task_fewshot = num_fewshot
                else:
                    # Use registry default
                    task_info = TASK_REGISTRY.get(task_name, {})
                    task_fewshot = task_info.get('default_fewshot', 0)
                
                task_limit = task_cfg.get('limit', limit)
                task_batch = task_cfg.get('batch_size', batch_size)
                
                logger.info(f"Evaluating {task_name} ({task_fewshot}-shot)")
                
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
                if 'results' in results and task_name in results['results']:
                    task_results = results['results'][task_name]
                    score = get_metric_from_results(task_results, task_name)
                    
                    if score is not None:
                        all_metrics[task_name] = score
                        logger.info(f"  ✓ {task_name}: {score:.4f} ({score*100:.2f}%)")
                    else:
                        logger.warning(f"  ✗ {task_name}: No valid metric found")
                else:
                    logger.warning(f"  ✗ {task_name}: No results returned")
                
            except Exception as e:
                logger.error(f"  ✗ {task_name} failed: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    traceback.print_exc()
                continue
        
        if not all_metrics:
            logger.warning("No metrics extracted from lm-eval results")
        else:
            logger.info(f"Successfully evaluated {len(all_metrics)}/{len(task_list)} tasks")
        
        return all_metrics
        
    except ImportError as e:
        logger.error(f"lm-eval not installed or incompatible: {e}")
        logger.error("Install with: pip install lm-eval")
        return {}
    except Exception as e:
        logger.error(f"Error running lm-eval-harness: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        return {}


def list_available_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Get list of all supported tasks with full info.
    
    Returns:
        Dictionary mapping task names to task info
    """
    return TASK_REGISTRY.copy()


def get_task_info(task_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Task information dict or None if not found
    """
    return TASK_REGISTRY.get(task_name)


def get_tasks_by_category(category: str) -> List[str]:
    """
    Get all tasks in a category.
    
    Args:
        category: Category name ('commonsense', 'knowledge', 'math', etc.)
        
    Returns:
        List of task names in that category
    """
    return [
        task_name for task_name, info in TASK_REGISTRY.items()
        if info.get('category') == category
    ]


def get_mistral_baseline_tasks() -> List[str]:
    """
    Get the tasks used in the Mistral 7B baseline evaluation.
    
    Returns:
        List of task names
    """
    return [
        # Commonsense (0-shot)
        'hellaswag', 'winogrande', 'piqa', 'siqa', 
        'openbookqa', 'arc_easy', 'arc_challenge', 'commonsense_qa',
        # Knowledge (5-shot)
        'nq_open', 'triviaqa',
        # Reading (0-shot)
        'boolq', 'quac',
        # Math
        'gsm8k', 'hendrycks_math',
        # Code
        'humaneval', 'mbpp',
        # Aggregate
        'mmlu', 'bbh', 'agieval'
    ]