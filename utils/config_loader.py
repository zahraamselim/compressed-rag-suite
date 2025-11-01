"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from JSON file."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config JSON file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self.config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG pipeline configuration."""
        return self.config.get('rag', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_efficiency_config(self) -> Dict[str, Any]:
        """Get efficiency evaluation configuration."""
        return self.config.get('evaluation', {}).get('efficiency', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance evaluation configuration."""
        return self.config.get('evaluation', {}).get('performance', {})
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval evaluation configuration."""
        return self.config.get('evaluation', {}).get('retrieval', {})
    
    def update_config(self, updates: Dict[str, Any], save: bool = False):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates (supports nested keys with dots)
            save: Whether to save changes to file
        """
        for key, value in updates.items():
            keys = key.split('.')
            current = self.config
            
            # Navigate to the nested location
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
        
        if save:
            self.save_config()
    
    def save_config(self, output_path: str = None):
        """
        Save configuration to file.
        
        Args:
            output_path: Output path (uses original path if None)
        """
        output_path = output_path or self.config_path
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
