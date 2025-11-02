"""Logging configuration utilities."""

import logging
import sys
import warnings
from typing import Optional, List


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    suppress_libraries: Optional[List[str]] = None,
    suppress_warnings: bool = True
):
    """
    Setup unified logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_string: Custom format string (uses default if None)
        suppress_libraries: List of library names to suppress (set to WARNING)
        suppress_warnings: Whether to suppress Python warnings
        
    Example:
        >>> from utils.logging_setup import setup_logging
        >>> setup_logging(level='INFO')
        >>> 
        >>> # With custom suppression
        >>> setup_logging(
        ...     level='DEBUG',
        ...     suppress_libraries=['transformers', 'torch']
        ... )
    """
    # Clear existing handlers
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    
    # Set format
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    
    # Add handler to root logger
    root.addHandler(handler)
    
    # Set level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    root.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # Suppress noisy libraries
    default_suppress = [
        'chromadb',
        'sentence_transformers',
        'transformers',
        'urllib3',
        'httpx',
        'httpcore',
        'filelock',
        'huggingface_hub',
        'torch.distributed'
    ]
    
    if suppress_libraries is not None:
        libraries_to_suppress = suppress_libraries
    else:
        libraries_to_suppress = default_suppress
    
    for lib in libraries_to_suppress:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Suppress Python warnings
    if suppress_warnings:
        warnings.filterwarnings('ignore')


def setup_simple_logging(verbose: bool = True):
    """
    Simple logging setup with minimal configuration.
    
    Args:
        verbose: If True, use INFO level. If False, use WARNING level.
        
    Example:
        >>> from utils.logging_setup import setup_simple_logging
        >>> setup_simple_logging(verbose=True)
    """
    level = "INFO" if verbose else "WARNING"
    setup_logging(level=level)


def setup_notebook_logging(verbose: bool = True):
    """
    Optimized logging for Jupyter notebooks.
    
    - Cleaner format
    - Suppresses more libraries
    - Shows progress bars properly
    
    Args:
        verbose: If True, use INFO level. If False, use WARNING level.
        
    Example:
        >>> from utils.logging_setup import setup_notebook_logging
        >>> setup_notebook_logging(verbose=True)
    """
    level = "INFO" if verbose else "WARNING"
    
    # Cleaner format for notebooks
    format_string = '%(asctime)s | %(levelname)s | %(message)s'
    
    setup_logging(
        level=level,
        format_string=format_string,
        suppress_warnings=True
    )


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with optional level override.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional level override
        
    Returns:
        Logger instance
        
    Example:
        >>> from utils.logging_setup import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello!")
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    return logger


def disable_library_logging(*library_names: str):
    """
    Disable logging for specific libraries.
    
    Args:
        *library_names: Names of libraries to disable
        
    Example:
        >>> from utils.logging_setup import disable_library_logging
        >>> disable_library_logging('transformers', 'torch')
    """
    for lib in library_names:
        logging.getLogger(lib).setLevel(logging.CRITICAL)


def enable_debug_logging(*module_names: str):
    """
    Enable DEBUG level logging for specific modules.
    
    Args:
        *module_names: Names of modules to enable debug for
        
    Example:
        >>> from utils.logging_setup import enable_debug_logging
        >>> enable_debug_logging('evaluation.efficiency')
    """
    for module in module_names:
        logging.getLogger(module).setLevel(logging.DEBUG)


# Convenient presets
def setup_for_development():
    """Setup logging for development (verbose, with debug info)."""
    setup_logging(
        level='DEBUG',
        format_string='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
    )


def setup_for_production():
    """Setup logging for production (minimal, warnings only)."""
    setup_logging(
        level='WARNING',
        format_string='%(asctime)s | %(levelname)s | %(message)s',
        suppress_warnings=True
    )


def setup_for_benchmarking():
    """Setup logging optimized for benchmarking (clean, performance-focused)."""
    setup_logging(
        level='INFO',
        format_string='%(message)s',
        suppress_warnings=True
    )
    
    # Suppress everything except evaluation and models
    disable_library_logging(
        'chromadb', 'sentence_transformers', 'transformers',
        'urllib3', 'httpx', 'torch', 'accelerate'
    )