"""Abstract base class for all model implementations."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """
    Abstract base class for ALL model interactions.
    
    This is the ONLY place that knows about model internals.
    Everything else uses this interface.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = None
        self.model_type = None  # 'base' or 'instruct'
    
    @abstractmethod
    def load(self, model_path: str, **kwargs):
        """Load model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_loglikelihood(self, text: str, context: str = "") -> float:
        """
        Get log-likelihood of text given context.
        
        Critical for accuracy benchmarks (HellaSwag, PIQA, etc.)
        
        Args:
            text: Full text to evaluate
            context: Context (text = context + continuation)
            
        Returns:
            Log probability of continuation
        """
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        pass
    
    def get_model(self):
        """Get the underlying model object."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def get_device(self):
        """Get the device."""
        return self.device
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
        }
        
        # Calculate model size
        if hasattr(self.model, 'parameters'):
            param_size = sum(p.element_size() * p.numel() for p in self.model.parameters())
            info["size_gb"] = param_size / (1024**3)
            info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
        
        # GPU memory if available
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
        return info
    
    def get_lm_eval_model(self):
        """
        Get lm-eval compatible model.
        
        Override this if you need custom lm-eval integration.
        Default tries HFLM wrapper.
        """
        try:
            from lm_eval.models.huggingface import HFLM
            return HFLM(
                pretrained=self.model,
                tokenizer=self.tokenizer,
                batch_size=1
            )
        except Exception as e:
            logger.warning(f"Could not create HFLM wrapper: {e}")
            return None
    
    def supports_lm_eval(self) -> bool:
        """Check if model works with lm-eval harness."""
        return self.get_lm_eval_model() is not None


def create_model_interface(model_type: str = "huggingface") -> ModelInterface:
    """
    Factory function to create appropriate model interface.
    
    Args:
        model_type: Type of model interface ('huggingface', 'llamacpp', etc.)
        
    Returns:
        ModelInterface instance
    """
    if model_type.lower() == "huggingface":
        from models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
