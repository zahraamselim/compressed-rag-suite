# Compressed RAG Evaluation Suite

A comprehensive evaluation framework for assessing Retrieval-Augmented Generation (RAG) systems with support for model compression techniques (quantization, pruning, distillation).

## Features

### Complete RAG Pipeline

- **Document Processing**: PDF, TXT, Markdown support with cleaning
- **Smart Chunking**: Semantic, sentence-based, and fixed-size strategies
- **Vector Search**: ChromaDB-based efficient similarity search
- **Context Retrieval**: Re-ranking and diversity mechanisms
- **Answer Generation**: Model-agnostic generation with chat templates

### Comprehensive Evaluation Suite

#### 1. Efficiency Benchmarks

- **Latency**: ms/token generation time
- **TTFT**: Time to first token
- **Throughput**: tokens/second
- **Memory**: Peak memory usage and model size
- **FLOPs & MFU**: Computational efficiency metrics
- **Energy**: Per-token energy consumption estimates

#### 2. Performance Benchmarks

- **Perplexity**: Language modeling quality on WikiText
- **LM-Eval Tasks**:
  - Reasoning: HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande
  - Knowledge: MMLU, TriviaQA, Natural Questions
  - Language: LAMBADA, StoryCloze
  - Math: GSM8K, MATH
  - Code: HumanEval, MBPP

#### 3. Retrieval Benchmarks

- **Retrieval Quality**: Precision@K, Recall@K, F1@K, MRR, MAP
- **Context Relevance**: Token overlap between query and context
- **Answer Quality**: Exact Match, Token F1, ROUGE, BERTScore
- **Faithfulness**: How well answers stick to retrieved context
- **No-RAG Comparison**: Improvement over baseline

## Installation

```bash
# Clone the repository
git clone https://github.com/zahraamselim/compressed-rag-suite.git
cd compressed-rag-suite

# Install dependencies
pip install torch transformers sentence-transformers chromadb
pip install nltk PyPDF2 numpy

# Optional: Advanced metrics
pip install rouge-score bert-score

# Optional: Performance benchmarks
pip install lm-eval datasets
```

## Quick Start

### 1. Basic RAG Evaluation

```python
from utils.config_loader import ConfigLoader
from models.model_interface import create_model_interface
from rag.pipeline import RAGPipeline
from evaluation.runner import EvaluationRunner

# Load configuration
config = ConfigLoader('config.json').get_config()

# Initialize model
model_interface = create_model_interface('huggingface')
model_interface.load(
    model_path='meta-llama/Llama-2-7b-chat-hf',
    model_type='instruct',
    torch_dtype='float16'
)

# Setup RAG pipeline
pipeline = RAGPipeline(config['rag'])
pipeline.setup(model_interface)

# Index documents
pipeline.index_documents('path/to/document.pdf')

# Run comprehensive evaluation
runner = EvaluationRunner(
    model_interface=model_interface,
    config=config,
    rag_pipeline=pipeline
)

results = runner.run_all(save_results=True)
print(results)
```

### 2. Command Line Usage

```bash
# Index and query
python main.py --config config.json \
    --index document.pdf \
    --query "What is machine learning?"

# Run efficiency benchmarks only
python main.py --config config.json --eval-efficiency

# Run performance benchmarks only
python main.py --config config.json --eval-performance

# Run retrieval benchmarks with dataset
python main.py --config config.json \
    --eval-retrieval \
    --retrieval-dataset qa_dataset.json

# Run all benchmarks
python main.py --config config.json --evaluate --verbose
```

## Configuration

Edit `config.json` to customize evaluation:

```json
{
  "model": {
    "model_path": "meta-llama/Llama-2-7b-chat-hf",
    "model_type": "instruct",
    "interface_type": "huggingface",
    "torch_dtype": "float16",
    "device_map": "auto"
  },
  "rag": {
    "chunking": {
      "strategy": "semantic",
      "chunk_size": 512,
      "chunk_overlap": 50
    },
    "embedding": {
      "model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "device": "cuda"
    },
    "retrieval": {
      "top_k": 3,
      "similarity_threshold": 0.3
    }
  },
  "evaluation": {
    "efficiency": {
      "num_runs": 10,
      "max_new_tokens": 128
    },
    "performance": {
      "measure_perplexity": true,
      "run_lm_eval": true,
      "lm_eval_tasks": {
        "hellaswag": { "enabled": true },
        "arc_easy": { "enabled": true },
        "arc_challenge": { "enabled": true }
      }
    },
    "retrieval": {
      "measure_retrieval_quality": true,
      "compare_no_rag": true,
      "k_values": [1, 3, 5, 10]
    }
  }
}
```

## Use Cases

### Evaluating Compressed Models

Perfect for comparing different compression techniques:

```python
# Baseline FP16 model
baseline_results = runner.run_all()

# 4-bit quantized model
model_interface.load(
    model_path='TheBloke/Llama-2-7B-Chat-GPTQ',
    quantization='gptq'
)
quantized_results = runner.run_all()

# Compare
print(f"Size reduction: {baseline_results.efficiency['model_size_gb'] / quantized_results.efficiency['model_size_gb']:.2f}x")
print(f"Speedup: {baseline_results.efficiency['latency_ms_per_token'] / quantized_results.efficiency['latency_ms_per_token']:.2f}x")
print(f"Quality retention: {quantized_results.performance['average_accuracy'] / baseline_results.performance['average_accuracy']:.2%}")
```

### Kaggle Notebooks

See `notebooks/` directory for complete examples:

- `mistral_7b_fp16.ipynb`: Baseline FP16 evaluation
- `mistral_7b_4bit.ipynb`: 4-bit quantization evaluation
- `comparison.ipynb`: Side-by-side comparison

## Results Format

Results are saved in JSON format:

```json
{
  "efficiency": {
    "latency_ms_per_token": 45.3,
    "throughput_tokens_per_sec": 22.1,
    "peak_memory_mb": 13421.8,
    "model_size_gb": 13.5
  },
  "performance": {
    "perplexity": 5.67,
    "lm_eval_scores": {
      "hellaswag": 0.7845,
      "arc_easy": 0.8123,
      "arc_challenge": 0.5234
    },
    "average_accuracy": 0.7067
  },
  "retrieval": {
    "precision_at_3": 0.8333,
    "recall_at_3": 0.75,
    "f1_score": 0.7234,
    "exact_match": 0.45,
    "faithfulness": 0.8912,
    "f1_improvement": 0.1523
  }
}
```

## Project Structure

```
compressed-rag-suite/
├── config.json              # Main configuration
├── main.py                  # CLI entry point
├── models/                  # Model interfaces
│   ├── model_interface.py
│   └── huggingface_model.py
├── rag/                     # RAG pipeline
│   ├── pipeline.py
│   ├── document_processing.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── indexing.py
│   ├── retrieval.py
│   └── generation.py
├── evaluation/              # Evaluation framework
│   ├── runner.py
│   ├── base.py
│   ├── efficiency/
│   ├── performance/
│   └── retrieval/
└── utils/                   # Utilities
    └── config_loader.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{compressed_rag_suite_2024,
  title={Compressed RAG Evaluation Suite},
  author={Zahraa Selim},
  year={2024},
  url={https://github.com/zahraamselim/compressed-rag-suite}
}
```

## Troubleshooting

### ChromaDB Issues

```bash
pip install --upgrade chromadb
```

### CUDA Out of Memory

Reduce batch sizes in `config.json`:

```json
{
  "rag": {
    "embedding": { "batch_size": 16 }
  }
}
```

### LM-Eval Not Found

```bash
pip install lm-eval
```

## Contact

For questions or issues, please open a GitHub issue or contact zahraamselim@gamil.com
