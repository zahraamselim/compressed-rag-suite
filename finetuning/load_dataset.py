"""
Factory function for loading datasets by category.
"""

def load_dataset(category: str, dataset_name: str, config=None):
    """
    Factory function to load any dataset by category and name.
    
    Args:
        category: Dataset category ('code_generation', 'math_reasoning', etc.)
        dataset_name: Specific dataset name
        config: Dataset configuration
    
    Returns:
        Loaded dataset instance
    
    Example:
        >>> dataset = load_dataset('code_generation', 'mbpp', {'include_tests': True})
        >>> train, eval = dataset.load()
    """
    if category == 'code_generation':
        from finetuning.code_generation import load_code_dataset
        return load_code_dataset(dataset_name, config)
    else:
        raise NotImplementedError(f"Category '{category}' not yet implemented")