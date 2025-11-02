import torch
from typing import Optional, Any
import logging
from pathlib import Path

from models.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class AWQModel(ModelInterface):
    """
    AWQ quantized model using AutoAWQ library.
    
    AWQ: Activation-aware Weight Quantization
    - Better quality than GPTQ at same bit-width
    - Very fast inference
    - Requires pre-quantized weights
    """
    
    def load(
        self,
        model_path: str,
        fuse_layers: bool = True,
        safetensors: bool = True,
        trust_remote_code: bool = False,
        model_type: str = "instruct",
        **kwargs
    ):
        """
        Load AWQ quantized model.
        
        Args:
            model_path: HF hub model ID or local path (must be AWQ quantized)
            fuse_layers: Fuse layers for faster inference
            safetensors: Use safetensors format
            trust_remote_code: Trust remote code
            model_type: 'base' or 'instruct'
            **kwargs: Additional arguments
        """
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "autoawq not installed. Install with: pip install autoawq"
            )
        
        logger.info(f"Loading AWQ model from {model_path}")
        self.model_path = model_path
        self.model_type = model_type
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load AWQ model
            self.model = AutoAWQForCausalLM.from_quantized(
                model_path,
                fuse_layers=fuse_layers,
                safetensors=safetensors,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            
            self.device = str(next(self.model.parameters()).device)
            
            # Log info
            info = self.get_model_info()
            logger.info(f"AWQ model loaded successfully")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Device: {info['device']}")
            logger.info(f"  Quantization: AWQ 4-bit")
            
            if torch.cuda.is_available():
                logger.info(f"  GPU Memory Allocated: {info.get('gpu_memory_allocated_gb', 0):.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load AWQ model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        return_full_text: bool = False,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        if return_full_text:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        
        return response.strip()
    
    def get_loglikelihood(self, text: str, context: str = "") -> float:
        """Get log-likelihood of text given context."""
        context_ids = self.tokenizer.encode(context, add_special_tokens=True) if context else []
        full_ids = self.tokenizer.encode(text, add_special_tokens=True)
        continuation_start = len(context_ids)
        
        input_tensor = torch.tensor([full_ids]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        total_log_prob = 0.0
        for i in range(continuation_start, len(full_ids)):
            if i == 0:
                continue
            token_id = full_ids[i]
            token_log_prob = log_probs[0, i-1, token_id].item()
            total_log_prob += token_log_prob
        
        return total_log_prob
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass returning logits."""
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            return outputs.logits
