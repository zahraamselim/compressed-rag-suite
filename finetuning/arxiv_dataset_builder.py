"""
Utilities for creating custom datasets from ArXiv papers.

Usage:
    builder = ArXivDatasetBuilder('data/papers/')
    builder.process_papers()
    builder.generate_qa_pairs()
    builder.save('arxiv_cs_dataset.json')
"""

import PyPDF2
import re
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ArXivDatasetBuilder:
    """
    Build domain expertise datasets from ArXiv papers.
    
    Steps:
    1. Extract text from PDFs
    2. Chunk into sections
    3. Generate QA pairs (manual or automatic)
    4. Format as training data
    """
    
    def __init__(self, papers_dir: str, domain: str = 'cs'):
        self.papers_dir = Path(papers_dir)
        self.domain = domain
        self.papers = []
        self.qa_pairs = []
    
    def process_papers(self, max_papers: Optional[int] = None):
        """Extract text from PDF papers."""
        logger.info(f"Processing papers from {self.papers_dir}")
        
        pdf_files = list(self.papers_dir.glob('*.pdf'))
        if max_papers:
            pdf_files = pdf_files[:max_papers]
        
        for pdf_file in pdf_files:
            try:
                text = self._extract_pdf_text(pdf_file)
                sections = self._chunk_into_sections(text)
                
                self.papers.append({
                    'filename': pdf_file.name,
                    'text': text,
                    'sections': sections
                })
                
                logger.info(f"Processed: {pdf_file.name}")
            except Exception as e:
                logger.warning(f"Failed to process {pdf_file.name}: {e}")
        
        logger.info(f"Processed {len(self.papers)} papers")
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        text = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return '\n\n'.join(text)
    
    def _chunk_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split paper into sections."""
        # Simple section detection
        sections = []
        
        # Common section headers
        patterns = [
            r'^(Abstract|Introduction|Background|Methods|Results|Discussion|Conclusion).*$',
            r'^\d+\.?\s+(Abstract|Introduction|Background|Methods|Results|Discussion|Conclusion).*$'
        ]
        
        lines = text.split('\n')
        current_section = {'title': 'Unknown', 'content': []}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_header = False
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_section['content']:
                        sections.append({
                            'title': current_section['title'],
                            'text': '\n'.join(current_section['content'])
                        })
                    
                    # Start new section
                    current_section = {'title': line, 'content': []}
                    is_header = True
                    break
            
            if not is_header:
                current_section['content'].append(line)
        
        # Add last section
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'text': '\n'.join(current_section['content'])
            })
        
        return sections
    
    def generate_qa_pairs_manual(self) -> List[Dict[str, str]]:
        """
        Generate QA pairs manually (template-based).
        
        Creates questions like:
        - "Summarize the main contribution of this paper"
        - "What methodology is used in this research?"
        - "What are the key findings?"
        """
        qa_pairs = []
        
        question_templates = [
            ("What is the main contribution described in this section?", "summary"),
            ("Explain the methodology used in this research.", "methodology"),
            ("What are the key findings or results?", "results"),
            ("What problem does this research address?", "problem"),
            ("Describe the technical approach taken.", "technical"),
        ]
        
        for paper in self.papers:
            for section in paper['sections']:
                # Skip very short sections
                if len(section['text'].split()) < 50:
                    continue
                
                # Create QA pairs based on section
                for question_template, qa_type in question_templates:
                    if self._section_matches_type(section['title'], qa_type):
                        qa_pairs.append({
                            'instruction': question_template,
                            'input': f"Section: {section['title']}\n\n{section['text'][:500]}...",
                            'output': section['text'][:300],  # Use first part as answer
                            'metadata': {
                                'paper': paper['filename'],
                                'section': section['title'],
                                'type': qa_type
                            }
                        })
        
        self.qa_pairs = qa_pairs
        logger.info(f"Generated {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def _section_matches_type(self, section_title: str, qa_type: str) -> bool:
        """Check if section matches QA type."""
        section_lower = section_title.lower()
        
        type_keywords = {
            'summary': ['abstract', 'introduction', 'conclusion'],
            'methodology': ['method', 'approach', 'implementation'],
            'results': ['result', 'evaluation', 'experiment'],
            'problem': ['introduction', 'background', 'motivation'],
            'technical': ['method', 'algorithm', 'architecture']
        }
        
        keywords = type_keywords.get(qa_type, [])
        return any(kw in section_lower for kw in keywords)
    
    def save(self, output_path: str):
        """Save dataset to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset to {output_path}")