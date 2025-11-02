"""Perplexity calculation for language models."""

import torch
import numpy as np
from typing import List, Optional
import logging
from tqdm import tqdm

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
    Uses proper loss calculation with padding mask.
    """
    
    def __init__(self, model_interface):
        """
        Initialize perplexity evaluator.
        
        Args:
            model_interface: ModelInterface instance
        """
        self.model_interface = model_interface
        self.model = model_interface.get_model()
        self.tokenizer = model_interface.get_tokenizer()
        self.device = model_interface.get_device()
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def calculate(
        self,
        text_samples: Optional[List[str]] = None,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        num_samples: Optional[int] = 100,
        max_length: int = 512,
        stride: Optional[int] = None,
        batch_size: int = 1
    ) -> float:
        """
        Calculate perplexity on text samples or dataset.
        
        Args:
            text_samples: Custom text samples (if None, load from dataset)
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split
            num_samples: Number of samples to use (None = all)
            max_length: Maximum sequence length
            stride: Stride for sliding window (None = no sliding, use simple batching)
            batch_size: Batch size for processing
            
        Returns:
            Perplexity value
        """
        logger.info("Calculating perplexity...")
        logger.info(f"  Dataset: {dataset_name}/{dataset_config}")
        logger.info(f"  Max length: {max_length}, Stride: {stride}")
        
        texts = self._get_text_samples(
            text_samples, dataset_name, dataset_config, split, num_samples
        )
        
        if not texts:
            logger.warning("No valid text samples available")
            return float('inf')
        
        logger.info(f"Processing {len(texts)} text samples...")
        
        if stride is not None and stride > 0:
            perplexity = self._calculate_with_stride(texts, max_length, stride)
        else:
            perplexity = self._calculate_simple(texts, max_length, batch_size)
        
        logger.info(f"Perplexity: {perplexity:.4f}")
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
            logger.info(f"Loading {dataset_name}/{dataset_config} (split: {split})...")
            dataset = load_dataset(
                dataset_name, 
                dataset_config if dataset_config else None,
                split=split, 
                trust_remote_code=True
            )
            
            if num_samples and len(dataset) > num_samples:
                # Sample evenly across the dataset
                indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
                dataset = dataset.select(indices)
            
            # Try different common field names
            texts = []
            text_field = None
            
            # Detect text field
            sample = dataset[0]
            for field in ['text', 'sentence', 'content', 'document', 'article']:
                if field in sample:
                    text_field = field
                    break
            
            if not text_field:
                logger.error(f"No text field found. Available fields: {list(sample.keys())}")
                return []
            
            logger.info(f"Using text field: '{text_field}'")
            
            for item in dataset:
                text = item.get(text_field, '')
                # Filter out empty or very short texts
                if text and len(text.strip()) > 10:
                    texts.append(text.strip())
            
            if not texts:
                logger.warning(f"No valid text found in dataset")
                return []
            
            logger.info(f"Loaded {len(texts)} valid text samples")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            return []
    
    def _calculate_simple(
        self, 
        texts: List[str], 
        max_length: int,
        batch_size: int = 1
    ) -> float:
        """
        Calculate perplexity without sliding window.
        
        This is faster but may be less accurate for very long texts.
        """
        total_nll = 0.0  # negative log-likelihood
        total_tokens = 0
        num_processed = 0
        
        # Use tqdm for progress if available
        try:
            from tqdm import tqdm
            texts_iter = tqdm(texts, desc="Computing perplexity")
        except ImportError:
            texts_iter = texts
        
        with torch.inference_mode():
            for text in texts_iter:
                if not text or len(text.strip()) == 0:
                    continue
                
                try:
                    # Tokenize
                    encodings = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        add_special_tokens=True
                    )
                    
                    input_ids = encodings["input_ids"].to(self.device)
                    
                    # Skip if too short
                    if input_ids.size(1) < 2:
                        continue
                    
                    # Get logits from model
                    outputs = self.model(input_ids, labels=input_ids)
                    
                    # Use model's loss if available (handles everything correctly)
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        nll = outputs.loss.item() * (input_ids.size(1) - 1)
                        num_tokens = input_ids.size(1) - 1
                    else:
                        # Manual calculation
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        
                        # Shift for next-token prediction
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = input_ids[:, 1:].contiguous()
                        
                        # Calculate cross-entropy loss
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        losses = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Sum the losses
                        nll = losses.sum().item()
                        num_tokens = shift_labels.numel()
                    
                    total_nll += nll
                    total_tokens += num_tokens
                    num_processed += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM on sample, skipping...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        logger.debug(f"Skipping sample due to error: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Skipping sample due to error: {e}")
                    continue
        
        if total_tokens == 0:
            logger.warning("No tokens processed successfully")
            return float('inf')
        
        # Calculate perplexity
        avg_nll = total_nll / total_tokens
        perplexity = np.exp(avg_nll)
        
        logger.info(f"Processed {num_processed}/{len(texts)} samples successfully")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Average NLL: {avg_nll:.4f}")
        
        return perplexity
    
    def _calculate_with_stride(
        self, 
        texts: List[str], 
        max_length: int, 
        stride: int
    ) -> float:
        """
        Calculate perplexity with sliding window.
        
        This is more accurate for long texts but slower.
        Uses stride to create overlapping windows.
        """
        total_nll = 0.0
        total_tokens = 0
        num_processed = 0
        
        logger.info(f"Using sliding window with stride {stride}")
        
        # Use tqdm for progress if available
        try:
            from tqdm import tqdm
            texts_iter = tqdm(texts, desc="Computing perplexity (stride)")
        except ImportError:
            texts_iter = texts
        
        with torch.inference_mode():
            for text in texts_iter:
                if not text or len(text.strip()) == 0:
                    continue
                
                try:
                    # Tokenize full text
                    encodings = self.tokenizer(
                        text, 
                        return_tensors="pt",
                        add_special_tokens=True
                    )
                    input_ids = encodings["input_ids"].to(self.device)
                    
                    seq_len = input_ids.size(1)
                    
                    if seq_len < 2:
                        continue
                    
                    # Process with sliding window
                    prev_end_loc = 0
                    for begin_loc in range(0, seq_len, stride):
                        end_loc = min(begin_loc + max_length, seq_len)
                        
                        # Extract chunk
                        input_chunk = input_ids[:, begin_loc:end_loc]
                        
                        if input_chunk.size(1) < 2:
                            continue
                        
                        # Get model outputs
                        outputs = self.model(input_chunk, labels=input_chunk)
                        
                        # Calculate loss for this chunk
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            # Only count the non-overlapping part
                            target_len = end_loc - begin_loc - (begin_loc - prev_end_loc) if begin_loc > 0 else end_loc - begin_loc - 1
                            nll = outputs.loss.item() * target_len
                            num_tokens = target_len
                        else:
                            # Manual calculation
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                            
                            # Shift for next-token prediction
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = input_chunk[:, 1:].contiguous()
                            
                            # For overlapping regions, only count new tokens
                            if begin_loc > 0:
                                # Skip overlapping part
                                overlap = begin_loc - prev_end_loc
                                if overlap < shift_labels.size(1):
                                    shift_logits = shift_logits[:, overlap:, :]
                                    shift_labels = shift_labels[:, overlap:]
                            
                            if shift_labels.numel() == 0:
                                prev_end_loc = end_loc
                                continue
                            
                            # Calculate loss
                            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(
                                shift_logits.reshape(-1, shift_logits.size(-1)),
                                shift_labels.reshape(-1)
                            )
                            
                            nll = losses.sum().item()
                            num_tokens = shift_labels.numel()
                        
                        total_nll += nll
                        total_tokens += num_tokens
                        prev_end_loc = end_loc
                        
                        if end_loc == seq_len:
                            break
                    
                    num_processed += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM on sample, skipping...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        logger.debug(f"Skipping sample due to error: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Skipping sample due to error: {e}")
                    continue
        
        if total_tokens == 0:
            logger.warning("No tokens processed successfully")
            return float('inf')
        
        # Calculate perplexity
        avg_nll = total_nll / total_tokens
        perplexity = np.exp(avg_nll)
        
        logger.info(f"Processed {num_processed}/{len(texts)} samples successfully")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Average NLL: {avg_nll:.4f}")
        
        return perplexity