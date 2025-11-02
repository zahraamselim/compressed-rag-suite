"""Convert PDF + QA pairs into evaluation-ready format."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import argparse

logger = logging.getLogger(__name__)


def load_qa_pairs(qa_file: str) -> List[Dict[str, str]]:
    """
    Load QA pairs from JSON file.
    
    Expected formats:
    1. [{"question": "...", "answer": "..."}, ...]
    2. {"questions": [...], "answers": [...]}
    """
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Format 1: List of dicts
        questions = [item['question'] for item in data]
        answers = [item.get('answer', item.get('ground_truth', '')) for item in data]
    elif isinstance(data, dict):
        # Format 2: Dict with lists
        questions = data.get('questions', [])
        answers = data.get('answers', data.get('ground_truth_answers', []))
    else:
        raise ValueError("Invalid QA file format")
    
    if len(questions) != len(answers):
        raise ValueError(f"Questions ({len(questions)}) and answers ({len(answers)}) count mismatch")
    
    return [{'question': q, 'answer': a} for q, a in zip(questions, answers)]


def extract_document_from_pdf(pdf_file: str) -> str:
    """
    Extract text from PDF.
    
    Returns single string with entire document.
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 required. Install: pip install PyPDF2")
    
    logger.info(f"Extracting text from {pdf_file}...")
    
    full_text = []
    with open(pdf_file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                full_text.append(text.strip())
    
    document = '\n\n'.join(full_text)
    logger.info(f"Extracted {len(document)} characters from {len(full_text)} pages")
    
    return document


def create_evaluation_dataset(
    qa_pairs: List[Dict[str, str]],
    document: Optional[str] = None,
    output_file: str = 'evaluation_dataset.json'
) -> Dict:
    """
    Create evaluation dataset in the format expected by RetrievalBenchmark.
    
    Args:
        qa_pairs: List of {"question": ..., "answer": ...}
        document: Full document text (optional, can be provided separately)
        output_file: Where to save
    
    Returns:
        Dataset dict
    """
    dataset = {
        'questions': [pair['question'] for pair in qa_pairs],
        'ground_truth_answers': [pair['answer'] for pair in qa_pairs],
    }
    
    if document:
        # Option 1: Provide full document (pipeline will chunk it)
        dataset['documents'] = [document]
    
    logger.info(f"Created dataset with {len(dataset['questions'])} question-answer pairs")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved dataset to {output_file}")
    
    return dataset


def main():
    """CLI for dataset conversion."""
    parser = argparse.ArgumentParser(
        description='Convert PDF + QA pairs to evaluation dataset'
    )
    parser.add_argument('--qa-file', required=True, help='QA pairs JSON file')
    parser.add_argument('--pdf-file', help='PDF document (optional)')
    parser.add_argument('--output', default='evaluation_dataset.json', 
                       help='Output file')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(args.qa_file)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Extract PDF if provided
    document = None
    if args.pdf_file:
        document = extract_document_from_pdf(args.pdf_file)
    
    # Create dataset
    dataset = create_evaluation_dataset(
        qa_pairs=qa_pairs,
        document=document,
        output_file=args.output
    )
    
    print(f"\n✓ Dataset created: {args.output}")
    print(f"  Questions: {len(dataset['questions'])}")
    print(f"  Documents: {len(dataset.get('documents', []))}")
    print(f"\nUsage:")
    print(f"  python main.py --config config.json \\")
    print(f"      --eval-retrieval \\")
    print(f"      --retrieval-dataset {args.output}")


if __name__ == '__main__':
    main()