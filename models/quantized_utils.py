"""
Utility module for loading and evaluating quantized models.
Place in: models/quantized_utils.py
"""

import logging
import sys
import warnings
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QuantizedModelConfig:
    """Configuration for quantized model loading."""
    model_name_or_path: str
    quantization_type: str  # 'gptq', 'awq', 'hqq', 'nf4'
    results_dir: Path
    device: str = 'cuda'
    
    # GPTQ specific
    use_triton: bool = False
    inject_fused_attention: bool = False
    inject_fused_mlp: bool = False
    
    # AWQ specific
    fuse_layers: bool = True
    
    # HQQ specific
    nbits: int = 4
    group_size: int = 64
    save_dir: Optional[str] = None
    
    # NF4 specific
    use_double_quant: bool = True
    compute_dtype: torch.dtype = torch.bfloat16


class QuantizedModelLoader:
    """Unified loader for different quantization methods."""
    
    @staticmethod
    def setup_logging(verbose: bool = True):
        """Setup unified logging configuration."""
        root = logging.getLogger()
        if root.handlers:
            root.handlers.clear()

        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s', 
            '%H:%M:%S'
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(logging.INFO if verbose else logging.WARNING)

        # Suppress noisy libraries
        for lib in ['chromadb', 'sentence_transformers', 'transformers', 
                    'urllib3', 'httpx']:
            logging.getLogger(lib).setLevel(logging.WARNING)

        warnings.filterwarnings('ignore')
        
    @staticmethod
    def load_gptq_model(config: QuantizedModelConfig) -> Tuple[Any, Any, float]:
        """Load GPTQ quantized model."""
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
        
        logger.info(f"Loading GPTQ model from {config.model_name_or_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, 
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoGPTQForCausalLM.from_quantized(
            config.model_name_or_path,
            device=f'{config.device}:0',
            use_safetensors=True,
            use_triton=config.use_triton,
            inject_fused_attention=config.inject_fused_attention,
            inject_fused_mlp=config.inject_fused_mlp
        )

        vram_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"✓ GPTQ model loaded | VRAM: {vram_used:.2f} GB")
        
        return model, tokenizer, vram_used
    
    @staticmethod
    def load_awq_model(config: QuantizedModelConfig) -> Tuple[Any, Any, float]:
        """Load AWQ quantized model."""
        from transformers import AutoTokenizer
        from awq import AutoAWQForCausalLM
        
        logger.info(f"Loading AWQ model from {config.model_name_or_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoAWQForCausalLM.from_quantized(
            config.model_name_or_path,
            fuse_layers=config.fuse_layers,
            safetensors=True
        )

        vram_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"✓ AWQ model loaded | VRAM: {vram_used:.2f} GB")
        
        return model, tokenizer, vram_used
    
    @staticmethod
    def load_hqq_model(config: QuantizedModelConfig) -> Tuple[Any, Any, float]:
        """Load or quantize HQQ model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from hqq.core.quantize import BaseQuantizeConfig
        from hqq.models.hf.base import AutoHQQHFModel
        
        logger.info("Loading/Quantizing HQQ model...")
        
        save_dir = config.save_dir or './model-hqq-4bit'
        
        if Path(save_dir).exists():
            logger.info(f"Loading pre-quantized HQQ model from {save_dir}")
            model, tokenizer = AutoHQQHFModel.from_quantized(save_dir)
            model = model.to(config.device).eval()
        else:
            logger.info("Quantizing model (may take 5-10 minutes)...")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
            
            quant_config = BaseQuantizeConfig(
                nbits=config.nbits, 
                group_size=config.group_size, 
                axis=1
            )
            AutoHQQHFModel.quantize_model(
                model, 
                quant_config=quant_config,
                compute_dtype=torch.float16, 
                device=config.device
            )
            
            AutoHQQHFModel.save_quantized(model, save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info(f"HQQ model saved to {save_dir}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        vram_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"✓ HQQ model loaded | VRAM: {vram_used:.2f} GB")
        
        return model, tokenizer, vram_used
    
    @staticmethod
    def load_nf4_model(config: QuantizedModelConfig) -> Tuple[Any, Any, float]:
        """Load NF4 quantized model using BitsAndBytes."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        logger.info(f"Loading NF4 model from {config.model_name_or_path}...")
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=config.use_double_quant,
            bnb_4bit_compute_dtype=config.compute_dtype
        )
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=nf4_config,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
        vram_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"✓ NF4 model loaded | VRAM: {vram_used:.2f} GB")
        
        return model, tokenizer, vram_used
    
    @classmethod
    def load_model(cls, config: QuantizedModelConfig) -> Tuple[Any, Any, float]:
        """Load model based on quantization type."""
        loaders = {
            'gptq': cls.load_gptq_model,
            'awq': cls.load_awq_model,
            'hqq': cls.load_hqq_model,
            'nf4': cls.load_nf4_model,
        }
        
        if config.quantization_type not in loaders:
            raise ValueError(
                f"Unknown quantization type: {config.quantization_type}. "
                f"Supported: {list(loaders.keys())}"
            )
        
        return loaders[config.quantization_type](config)


class EvaluationConfigBuilder:
    """Builder for creating evaluation configurations."""
    
    @staticmethod
    def create_comprehensive_config(
        model_path: str,
        results_dir: Path,
        enable_rag: bool = True,
        enable_performance: bool = True,
        enable_efficiency: bool = True,
        lm_eval_limit: int = 100,
        num_rag_questions: int = 10
    ) -> Dict[str, Any]:
        """Create a comprehensive evaluation configuration."""
        config = {
            'model.model_path': model_path,
            'model.model_type': 'instruct',
            'evaluation.output_dir': str(results_dir),
        }
        
        # Efficiency benchmarks
        if enable_efficiency:
            config.update({
                'evaluation.efficiency.enabled': True,
                'evaluation.efficiency.num_runs': 5,
                'evaluation.efficiency.latency.enabled': True,
                'evaluation.efficiency.throughput.enabled': True,
                'evaluation.efficiency.memory.enabled': True,
                'evaluation.efficiency.energy.enabled': True,
                'evaluation.efficiency.flops.enabled': True,
            })
        
        # Performance benchmarks
        if enable_performance:
            config.update({
                'evaluation.performance.enabled': True,
                'evaluation.performance.perplexity.enabled': False,
                'evaluation.performance.lm_eval.enabled': True,
                'evaluation.performance.lm_eval.batch_size': 1,
                'evaluation.performance.lm_eval.tasks.hellaswag.enabled': True,
                'evaluation.performance.lm_eval.tasks.hellaswag.limit': lm_eval_limit,
                'evaluation.performance.lm_eval.tasks.piqa.enabled': True,
                'evaluation.performance.lm_eval.tasks.piqa.limit': lm_eval_limit,
                'evaluation.performance.lm_eval.tasks.arc_easy.enabled': True,
                'evaluation.performance.lm_eval.tasks.arc_easy.limit': lm_eval_limit,
                'evaluation.performance.lm_eval.tasks.winogrande.enabled': True,
                'evaluation.performance.lm_eval.tasks.winogrande.limit': lm_eval_limit,
            })
        
        # RAG evaluation
        if enable_rag:
            config.update({
                'evaluation.retrieval.enabled': True,
                'evaluation.retrieval.document_path': './data/2308.07633v4-clean.pdf',
                'evaluation.retrieval.qa_pairs_path': './data/2308.07633v4-qa.json',
                'evaluation.retrieval.num_questions': num_rag_questions,
                'evaluation.retrieval.compare_no_rag': True,
                
                # RAG pipeline config
                'rag.chunking.chunk_size': 512,
                'rag.chunking.chunk_overlap': 50,
                'rag.chunking.strategy': 'semantic',
                'rag.embedding.model_name': 'all-MiniLM-L6-v2',
                'rag.embedding.device': 'cuda',
                'rag.retrieval.top_k': 3,
                'rag.retrieval.similarity_threshold': 0.3,
                'rag.generation.max_new_tokens': 150,
                'rag.generation.temperature': 0.3,
            })
        
        return config


def print_evaluation_summary(
    results: Any, 
    model_info: Dict[str, Any], 
    vram_used: float,
    quantization_type: str
):
    """Print comprehensive evaluation summary."""
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    
    if results.efficiency:
        logger.info("\n📊 EFFICIENCY METRICS:")
        logger.info(f"  Latency: {results.efficiency.latency.mean_ms:.2f}ms "
                   f"± {results.efficiency.latency.std_ms:.2f}ms")
        logger.info(f"  Throughput: {results.efficiency.throughput.tokens_per_second:.2f} tokens/s")
        logger.info(f"  Memory: {results.efficiency.memory.peak_allocated_gb:.2f} GB")
        if results.efficiency.energy:
            logger.info(f"  Energy: {results.efficiency.energy.total_joules:.2f} J")
        if results.efficiency.flops:
            logger.info(f"  FLOPs: {results.efficiency.flops.total_flops/1e9:.2f} GFLOPs")
    
    if results.performance:
        logger.info("\n📈 PERFORMANCE METRICS:")
        if results.performance.lm_eval_scores:
            for task, score in sorted(results.performance.lm_eval_scores.items()):
                logger.info(f"  {task}: {score*100:.2f}%")
            logger.info(f"  Average: {results.performance.average_accuracy*100:.2f}%")
    
    if results.retrieval:
        logger.info("\n🔍 RAG METRICS:")
        logger.info(f"  F1 Score (RAG): {results.retrieval.f1_score:.4f}")
        logger.info(f"  Exact Match (RAG): {results.retrieval.exact_match:.4f}")
        logger.info(f"  Answer Relevance: {results.retrieval.answer_relevance:.4f}")
        logger.info(f"  Context Precision: {results.retrieval.context_precision:.4f}")
        logger.info(f"  Context Recall: {results.retrieval.context_recall:.4f}")
        if results.retrieval.no_rag_f1:
            logger.info(f"\n  F1 Score (No-RAG): {results.retrieval.no_rag_f1:.4f}")
            improvement = (results.retrieval.f1_score - results.retrieval.no_rag_f1) * 100
            logger.info(f"  F1 Improvement: {improvement:+.2f}%")
    
    logger.info("\n" + "="*80)


# Example usage in notebook:
"""
from models.quantized_utils import (
    QuantizedModelLoader, 
    QuantizedModelConfig,
    EvaluationConfigBuilder,
    print_evaluation_summary
)

# Setup
QuantizedModelLoader.setup_logging()

# Load model
config = QuantizedModelConfig(
    model_name_or_path='TheBloke/Mistral-7B-Instruct-v0.1-GPTQ',
    quantization_type='gptq',
    results_dir=Path('./results/mistral_7b_gptq')
)

model, tokenizer, vram_used = QuantizedModelLoader.load_model(config)

# Create evaluation config
eval_config = EvaluationConfigBuilder.create_comprehensive_config(
    model_path=config.model_name_or_path,
    results_dir=config.results_dir,
    lm_eval_limit=100,
    num_rag_questions=10
)

# Run evaluation
from evaluation.runner import EvaluationRunner
from config_loader import ConfigLoader

config_loader = ConfigLoader('./config.json')
config_loader.update_config(eval_config)

eval_runner = EvaluationRunner(model_interface=model_interface, ...)
results = eval_runner.run_all()

# Print summary
print_evaluation_summary(results, info, vram_used, 'GPTQ')
"""
