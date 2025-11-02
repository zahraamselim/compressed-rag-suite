import torch
from typing import Optional, Any
import logging
from pathlib import Path

from models.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class HQQModel(ModelInterface):
    """
    HQQ quantized model using HQQ library.
    
    HQQ: Half-Quadratic Quantization
    - On-the-fly quantization (no pre-quantized weights needed)
    - Flexible bit-widths (2, 3, 4, 8-bit)
    - Good quality/speed tradeoff
    """
    
    def load(
        self,
        model_path: str,
        nbits: int = 4,
        group_size: int = 64,
        axis: int = 1,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        load_quantized: bool = False,
        quantized_path: Optional[str] = None,
        trust_remote_code: bool = False,
        model_type: str = "instruct",
        **kwargs
    ):
        """
        Load or quantize model with HQQ.
        
        Args:
            model_path: HF hub model ID or local path
            nbits: Number of bits (2, 3, 4, 8)
            group_size: Group size for quantization
            axis: Quantization axis (0 or 1)
            compute_dtype: Compute dtype
            device: Device to load on
            load_quantized: Load pre-quantized HQQ model
            quantized_path: Path to pre-quantized model
            trust_remote_code: Trust remote code
            model_type: 'base' or 'instruct'
            **kwargs: Additional arguments
        """
        try:
            from hqq.core.quantize import BaseQuantizeConfig
            from hqq.models.hf.base import AutoHQQHFModel
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "hqq not installed. Install with: pip install hqq"
            )
        
        logger.info(f"Loading HQQ model from {model_path}")
        self.model_path = model_path
        self.model_type = model_type
        self.nbits = nbits
        
        try:
            if load_quantized and quantized_path and Path(quantized_path).exists():
                # Load pre-quantized model
                logger.info(f"Loading pre-quantized HQQ model from {quantized_path}")
                self.model, self.tokenizer = AutoHQQHFModel.from_quantized(quantized_path)
                self.model = self.model.to(device).eval()
            else:
                # Load and quantize on-the-fly
                logger.info(f"Quantizing model to {nbits}-bit (may take 5-10 minutes)...")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code
                )
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=trust_remote_code
                )
                
                # Quantize
                quant_config = BaseQuantizeConfig(
                    nbits=nbits,
                    group_size=group_size,
                    axis=axis
                )
                
                AutoHQQHFModel.quantize_model(
                    base_model,
                    quant_config=quant_config,
                    compute_dtype=compute_dtype,
                    device=device
                )
                
                self.model = base_model
                
                # Save if path provided
                if quantized_path:
                    logger.info(f"Saving quantized model to {quantized_path}")
                    Path(quantized_path).mkdir(parents=True, exist_ok=True)
                    AutoHQQHFModel.save_quantized(self.model, quantized_path)
                    self.tokenizer.save_pretrained(quantized_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.device = device
            
            # Log info
            info = self.get_model_info()
            logger.info(f"HQQ model loaded successfully")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Device: {info['device']}")
            logger.info(f"  Quantization: HQQ {nbits}-bit")
            
            if torch.cuda.is_available():
                logger.info(f"  GPU Memory Allocated: {info.get('gpu_memory_allocated_gb', 0):.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load HQQ model: {e}")
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
    
    def get_model_info(self) -> dict:
        """Get model information including quantization details."""
        info = super().get_model_info()
        info['quantization'] = 'HQQ'
        info['bits_per_param'] = self.nbits
        return info