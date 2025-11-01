"""Perplexity calculation for language models."""

import torch
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Check datasets availability
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not available. Install with: pip install datasets")


class PerplexityEvaluator:
    """
    Evaluate language model perplexity.
    
    Supports custom text samples or standard datasets.
    """
    
    def __init__(self, model_interface):
        """
        Initialize perplexity evaluator.
        
        Args:
            model_interface: ModelInterface instance
        """
        self.model_interface = model_interface
        self.device = model_interface.get_device()
    
    def calculate(
        self,
        text_samples: Optional[List[str]] = None,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        num_samples: Optional[int] = 100,
        max_length: int = 512,
        stride: Optional[int] = None
    ) -> float:
        """
        Calculate perplexity on text samples or dataset.
        
        Args:
            text_samples: Custom text samples (if None, load from dataset)
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split
            num_samples: Number of samples to use
            max_length: Maximum sequence length
            stride: Stride for sliding window (None = no sliding)
            
        Returns:
            Perplexity value
        """
        logger.info("Calculating perplexity...")
        
        texts = self._get_text_samples(
            text_samples, dataset_name, dataset_config, split, num_samples
        )
        
        if not texts:
            logger.warning("No valid text samples available")
            return float('inf')
        
        if stride is not None:
            perplexity = self._calculate_with_stride(texts, max_length, stride)
        else:
            perplexity = self._calculate_simple(texts, max_length)
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity
    
    def _get_text_samples(
        self,
        text_samples: Optional[List[str]],
        dataset_name: str,
        dataset_config: str,
        split: str,
        num_samples: Optional[int]
    ) -> List[str]:
        """Load text samples from provided list or dataset."""
        if text_samples:
            logger.info(f"Using {len(text_samples)} provided text samples")
            return text_samples
        
        if not DATASETS_AVAILABLE:
            logger.error("datasets library not available. Provide text_samples or install: pip install datasets")
            return []
        
        try:
            logger.info(f"Loading {dataset_name}/{dataset_config}...")
            dataset = load_dataset(
                dataset_name, dataset_config, split=split, trust_remote_code=True
            )
            
            if num_samples and len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
            
            # Try different common field names
            texts = []
            for item in dataset:
                text = item.get('text') or item.get('sentence') or item.get('content') or ''
                if text and len(text.strip()) > 0:
                    texts.append(text)
            
            if not texts:
                logger.warning(f"No valid text found in dataset. Available fields: {list(dataset[0].keys())}")
                return []
            
            logger.info(f"Loaded {len(texts)} text samples")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []
    
    def _calculate_simple(self, texts: List[str], max_length: int) -> float:
        """Calculate perplexity without sliding window."""
        total_loss = 0.0
        total_tokens = 0
        tokenizer = self.model_interface.get_tokenizer()
        
        with torch.no_grad():
            for text in texts:
                if not text or len(text.strip()) == 0:
                    continue
                
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length
                    )
                    
                    if inputs["input_ids"].size(1) < 2:
                        continue
                    
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    logits = self.model_interface.forward(inputs["input_ids"])
                    
                    # Shift for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    
                    # Calculate cross entropy loss
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    total_loss += loss.item()
                    total_tokens += shift_labels.numel()
                    
                except Exception as e:
                    logger.debug(f"Skipping sample due to error: {e}")
                    continue
        
        if total_tokens == 0:
            logger.warning("No tokens processed successfully")
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def _calculate_with_stride(
        self, texts: List[str], max_length: int, stride: int
    ) -> float:
        """Calculate perplexity with sliding window (more accurate for long texts)."""
        total_loss = 0.0
        total_tokens = 0
        tokenizer = self.model_interface.get_tokenizer()
        
        with torch.no_grad():
            for text in texts:
                if not text or len(text.strip()) == 0:
                    continue
                
                try:
                    encodings = tokenizer(text, return_tensors="pt")
                    input_ids = encodings["input_ids"].to(self.device)
                    
                    seq_len = input_ids.size(1)
                    
                    # Sliding window
                    for begin_loc in range(0, seq_len, stride):
                        end_loc = min(begin_loc + max_length, seq_len)
                        
                        input_chunk = input_ids[:, begin_loc:end_loc]
                        
                        if input_chunk.size(1) < 2:
                            continue
                        
                        logits = self.model_interface.forward(input_chunk)
                        
                        # Shift for next-token prediction
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_chunk[..., 1:].contiguous()
                        
                        # Calculate cross entropy loss
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        total_loss += loss.item()
                        total_tokens += shift_labels.numel()
                        
                        if end_loc == seq_len:
                            break
                            
                except Exception as e:
                    logger.debug(f"Skipping chunk due to error: {e}")
                    continue
        
        if total_tokens == 0:
            logger.warning("No tokens processed successfully")
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
