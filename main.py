"""Main entry point for RAG evaluation system."""

import argparse
import logging
import torch
from pathlib import Path

from utils.config_loader import ConfigLoader
from models.model_interface import create_model_interface
from rag.pipeline import RAGPipeline
from evaluation.runner import EvaluationRunner 

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_torch_dtype(dtype_str: str):
    """Parse torch dtype from string."""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float64': torch.float64,
    }
    return dtype_map.get(dtype_str, torch.float16)

def should_retrieve(query: str, model_interface) -> bool:
    """Classify if query needs retrieval."""
    # Simple heuristic - check if query is about the document
    keywords = ['what', 'how', 'why', 'explain', 'describe', 'define']
    return any(kw in query.lower() for kw in keywords)

def main():
    parser = argparse.ArgumentParser(description='RAG Evaluation System')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--index', type=str, default=None,
                        help='Document to index (PDF or TXT)')
    parser.add_argument('--query', type=str, default=None,
                        help='Query to test')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation benchmarks')
    parser.add_argument('--eval-efficiency', action='store_true',
                        help='Run only efficiency benchmarks')
    parser.add_argument('--eval-performance', action='store_true',
                        help='Run only performance benchmarks')
    parser.add_argument('--eval-retrieval', action='store_true',
                        help='Run only retrieval benchmarks')
    parser.add_argument('--retrieval-dataset', type=str, default=None,
                        help='Path to retrieval evaluation dataset')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config_loader = ConfigLoader(args.config)
        config = config_loader.get_config()
        
        # Load model
        logger.info("Loading model...")
        model_config = config_loader.get_model_config()
        model_interface = create_model_interface(model_config.get('interface_type', 'huggingface'))
        model_interface.load(
            model_path=model_config['model_path'],
            model_type=model_config.get('model_type', 'instruct'),
            torch_dtype=parse_torch_dtype(model_config.get('torch_dtype', 'float16')),
            device_map=model_config.get('device_map', 'auto'),
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_config = config_loader.get_rag_config()
        pipeline = RAGPipeline(rag_config)
        pipeline.setup(model_interface)
        
        # Index document if provided
        if args.index:
            logger.info(f"Indexing document: {args.index}")
            processing_time = pipeline.index_documents(args.index)
            logger.info(f"Document indexed in {processing_time:.2f}s")
            logger.info(f"Stats: {pipeline.get_stats()}")
        
        # Test query if provided
        if args.query:
            logger.info(f"Query: {args.query}")
            
            if should_retrieve(args.query, model_interface):
                result = pipeline.query(args.query, return_context=True, return_chunks=True)
                
                print("\n" + "="*80)
                print("QUERY:", args.query)
                print("="*80)
                
                if 'chunks' in result and result['chunks']:
                    print("\nRETRIEVED CHUNKS:")
                    for i, chunk in enumerate(result['chunks'], 1):
                        print(f"\n[Chunk {i}] (score: {chunk['score']:.3f})")
                        print(chunk['text'][:200] + ("..." if len(chunk['text']) > 200 else ""))
                
                print("\n" + "="*80)
                print("ANSWER:")
                print(result['answer'])
                print("="*80 + "\n")
            else:
                # Direct generation without RAG
                answer = model_interface.generate(args.query, max_new_tokens=64)
                
                print("\n" + "="*80)
                print("QUERY:", args.query)
                print("="*80)
                print("\nMODE: Direct generation (no retrieval)")
                print("\n" + "="*80)
                print("ANSWER:")
                print(answer)
                print("="*80 + "\n")
        
        # Run evaluation if requested
        if args.evaluate or args.eval_efficiency or args.eval_performance or args.eval_retrieval:
            logger.info("Initializing evaluation runner...")
            
            runner = EvaluationRunner(
                model_interface=model_interface,
                config=config,
                rag_pipeline=pipeline,
                verbose=args.verbose
            )
            
            output_dir = config['evaluation'].get('output_dir', './results')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Determine what to run
            if args.evaluate:
                # Run all benchmarks
                logger.info("Running comprehensive evaluation...")
                results = runner.run_all(
                    save_results=True,
                    output_dir=output_dir
                )
                print(results)
            
            else:
                # Run specific benchmarks
                if args.eval_efficiency:
                    logger.info("Running efficiency benchmarks...")
                    eff_results = runner.run_efficiency_only()
                    eff_results.to_json(f"{output_dir}/efficiency_results.json")
                    print(eff_results)
                
                if args.eval_performance:
                    logger.info("Running performance benchmarks...")
                    perf_results = runner.run_performance_only()
                    perf_results.to_json(f"{output_dir}/performance_results.json")
                    print(perf_results)
                
                if args.eval_retrieval:
                    logger.info("Running retrieval benchmarks...")
                    
                    # Load dataset if provided
                    if args.retrieval_dataset:
                        from evaluation.retrieval import RetrievalBenchmark
                        retrieval_benchmark = RetrievalBenchmark(
                            model_interface=model_interface,
                            rag_pipeline=pipeline,
                            config=config['evaluation']['retrieval'],
                            verbose=args.verbose
                        )
                        ret_results = retrieval_benchmark.evaluate_from_file(args.retrieval_dataset)
                    else:
                        ret_results = runner.run_retrieval_only()
                    
                    ret_results.to_json(f"{output_dir}/retrieval_results.json")
                    print(ret_results)
            
            logger.info(f"Results saved to {output_dir}/")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
