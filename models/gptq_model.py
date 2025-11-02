"""GPTQ quantized model implementation."""

import torch
from typing import Optional
import logging

from models.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class GPTQModel(ModelInterface):
    """
    GPTQ 4-bit quantized model.
    
    Requires: auto-gptq
    pip install auto-gptq
    
    Works with models from TheBloke (e.g., TheBloke/Mistral-7B-Instruct-v0.1-GPTQ)
    """
    
    def load(
        self,
        model_path: str,
        device: str = "cuda:0",
        use_triton: bool = False,
        use_safetensors: bool = True,
        inject_fused_attention: bool = False,
        inject_fused_mlp: bool = False,
        trust_remote_code: bool = False,
        model_type: str = "instruct",
        **kwargs
    ):
        """
        Load GPTQ quantized model.
        
        Args:
            model_path: Path or HF hub model ID (must be GPTQ quantized)
            device: Device to load on
            use_triton: Use Triton kernels for faster inference
            use_safetensors: Use safetensors format
            inject_fused_attention: Use fused attention (faster but experimental)
            inject_fused_mlp: Use fused MLP (faster but experimental)
            trust_remote_code: Whether to trust remote code
            model_type: 'base' or 'instruct'
            **kwargs: Additional arguments for AutoGPTQForCausalLM
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "auto-gptq is required for GPTQ models. "
                "Install with: pip install auto-gptq"
            )
        
        logger.info(f"Loading GPTQ model from {model_path}")
        self.model_path = model_path
        self.model_type = model_type
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=trust_remote_code
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load GPTQ model
            self.model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                device=device,
                use_safetensors=use_safetensors,
                use_triton=use_triton,
                inject_fused_attention=inject_fused_attention,
                inject_fused_mlp=inject_fused_mlp,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            self.model.eval()
            
            # Set device
            self.device = device
            
            # Log model info
            info = self.get_model_info()
            logger.info(f"GPTQ model loaded successfully")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Size: {info.get('size_gb', 0):.2f} GB")
            logger.info(f"  Parameters: {info.get('num_parameters', 0):,}")
            logger.info(f"  Device: {info['device']}")
            
            if 'gpu_memory_allocated_gb' in info:
                logger.info(f"  GPU Memory: {info['gpu_memory_allocated_gb']:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load GPTQ model: {e}")
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
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            return_full_text: If True, return prompt + generation
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
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
        """
        Get log-likelihood of text given context.
        
        Args:
            text: Full text to evaluate
            context: Context prefix
            
        Returns:
            Log probability of the continuation
        """
        # Encode context and full text
        context_ids = self.tokenizer.encode(
            context, 
            add_special_tokens=True
        ) if context else []
        
        full_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Determine where continuation starts
        continuation_start = len(context_ids)
        
        # Create input tensor
        input_tensor = torch.tensor([full_ids]).to(self.device)
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits
        
        # Calculate log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Sum log probs for continuation tokens
        total_log_prob = 0.0
        for i in range(continuation_start, len(full_ids)):
            if i == 0:
                continue
            token_id = full_ids[i]
            token_log_prob = log_probs[0, i-1, token_id].item()
            total_log_prob += token_log_prob
        
        return total_log_prob
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            **kwargs: Additional forward pass arguments
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            return outputs.logits