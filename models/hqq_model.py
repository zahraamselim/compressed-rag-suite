"""HQQ quantized model implementation."""

import torch
from typing import Optional
from pathlib import Path
import logging

from models.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class HQQModel(ModelInterface):
    """
    HQQ (Half-Quadratic Quantization) model.
    
    Requires: hqq
    pip install hqq
    
    Supports 2/3/4/8-bit quantization with flexible group sizes.
    Can quantize on-the-fly or load pre-quantized models.
    """
    
    def load(
        self,
        model_path: str,
        nbits: int = 4,
        group_size: int = 64,
        axis: int = 1,
        device: str = "cuda",
        compute_dtype: torch.dtype = torch.float16,
        save_dir: Optional[str] = None,
        force_quantize: bool = False,
        trust_remote_code: bool = False,
        model_type: str = "instruct",
        **kwargs
    ):
        """
        Load or quantize HQQ model.
        
        Args:
            model_path: Path or HF hub model ID (FP16 model to quantize)
            nbits: Number of bits (2, 3, 4, or 8)
            group_size: Quantization group size
            axis: Quantization axis (0 or 1)
            device: Device to load on
            compute_dtype: Compute dtype for quantization
            save_dir: Directory to save/load quantized model
            force_quantize: Force re-quantization even if save_dir exists
            trust_remote_code: Whether to trust remote code
            model_type: 'base' or 'instruct'
            **kwargs: Additional arguments
        """
        try:
            from hqq.core.quantize import BaseQuantizeConfig
            from hqq.models.hf.base import AutoHQQHFModel
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "hqq is required for HQQ models. "
                "Install with: pip install hqq"
            )
        
        logger.info(f"Loading/Quantizing HQQ model from {model_path}")
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        
        try:
            # Determine save directory
            if save_dir is None:
                save_dir = f"./hqq_{nbits}bit_{Path(model_path).name}"
            
            save_path = Path(save_dir)
            
            # Check if pre-quantized model exists
            if save_path.exists() and not force_quantize:
                logger.info(f"Loading pre-quantized HQQ model from {save_dir}")
                
                # Load quantized model
                self.model, self.tokenizer = AutoHQQHFModel.from_quantized(str(save_path))
                self.model = self.model.to(device).eval()
                
            else:
                # Quantize model
                logger.info(f"Quantizing model to {nbits}-bit (may take 5-10 minutes)...")
                
                # Load FP16 model
                model_fp16 = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=trust_remote_code
                )
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code
                )
                
                # Create quantization config
                quant_config = BaseQuantizeConfig(
                    nbits=nbits,
                    group_size=group_size,
                    axis=axis
                )
                
                # Quantize
                logger.info(f"  Config: {nbits}-bit, group_size={group_size}, axis={axis}")
                AutoHQQHFModel.quantize_model(
                    model_fp16,
                    quant_config=quant_config,
                    compute_dtype=compute_dtype,
                    device=device
                )
                
                self.model = model_fp16
                
                # Save quantized model
                if save_dir:
                    logger.info(f"Saving quantized model to {save_dir}")
                    save_path.mkdir(parents=True, exist_ok=True)
                    AutoHQQHFModel.save_quantized(self.model, str(save_path))
                    self.tokenizer.save_pretrained(str(save_path))
                    logger.info(f"Model saved to {save_dir}")
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.model.eval()
            
            # Log model info
            info = self.get_model_info()
            logger.info(f"HQQ model loaded successfully")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Quantization: {nbits}-bit")
            logger.info(f"  Size: {info.get('size_gb', 0):.2f} GB")
            logger.info(f"  Parameters: {info.get('num_parameters', 0):,}")
            logger.info(f"  Device: {info['device']}")
            
            if 'gpu_memory_allocated_gb' in info:
                logger.info(f"  GPU Memory: {info['gpu_memory_allocated_gb']:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load/quantize HQQ model: {e}")
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