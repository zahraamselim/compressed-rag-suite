"""LLM answer generation with RAG."""

import logging
import re
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Generate answers using LLM with retrieved context.
    Uses ModelInterface for true model agnosticism.
    """
    
    def __init__(self, model_interface, config: dict):
        """
        Initialize RAG generator.
        
        Args:
            model_interface: ModelInterface instance
            config: Generation config from config.json
        """
        self.model_interface = model_interface
        
        self.max_new_tokens = config.get('max_new_tokens', 150)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.do_sample = config.get('do_sample', True)
        self.repetition_penalty = config.get('repetition_penalty', 1.1)
        
        self.model_type = model_interface.model_type or "instruct"
        self.use_chat_template = config.get('use_chat_template', True)
        
        logger.info(f"Initialized RAG generator for {self.model_type} model")
    
    def generate(
        self,
        query: str,
        context: str,
        return_prompt: bool = False
    ) -> Union[str, Tuple[str, str]]:
        """
        Generate answer given query and context.
        
        Args:
            query: User question
            context: Retrieved context
            return_prompt: If True, return (answer, prompt) tuple
            
        Returns:
            Generated answer (or tuple with prompt)
        """
        # Format prompt based on model type
        if self.model_type == "instruct":
            prompt = self._format_instruct_prompt(query, context)
        else:
            prompt = self._format_base_prompt(query, context)
        
        # Use model interface for generation
        answer = self.model_interface.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        
        answer = self._clean_answer(answer)
        
        if return_prompt:
            return answer, prompt
        return answer
    
    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate answers for multiple queries in batch.
        
        Note: Current implementation processes sequentially.
        True batch processing requires model-specific implementation.
        
        Args:
            queries: List of questions
            contexts: List of contexts (one per query)
            show_progress: Show progress indicator
            
        Returns:
            List of generated answers
        """
        if len(queries) != len(contexts):
            raise ValueError(f"Queries ({len(queries)}) and contexts ({len(contexts)}) must have same length")
        
        answers = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(zip(queries, contexts), total=len(queries), desc="Generating answers")
            except ImportError:
                iterator = zip(queries, contexts)
                logger.warning("tqdm not available, progress bar disabled")
        else:
            iterator = zip(queries, contexts)
        
        for query, context in iterator:
            answer = self.generate(query, context)
            answers.append(answer)
        
        return answers
    
    def generate_without_context(self, query: str) -> str:
        """
        Generate answer without RAG context.
        
        Args:
            query: User question
            
        Returns:
            Generated answer
        """
        if self.model_type == "instruct":
            prompt = self._format_instruct_prompt_no_rag(query)
        else:
            prompt = self._format_base_prompt_no_rag(query)
        
        answer = self.model_interface.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        
        return self._clean_answer(answer)
    
    def generate_batch_without_context(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate answers for multiple queries without context.
        
        Args:
            queries: List of questions
            show_progress: Show progress indicator
            
        Returns:
            List of generated answers
        """
        answers = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(queries, desc="Generating no-RAG answers")
            except ImportError:
                iterator = queries
                logger.warning("tqdm not available, progress bar disabled")
        else:
            iterator = queries
        
        for query in iterator:
            answer = self.generate_without_context(query)
            answers.append(answer)
        
        return answers
    
    def _format_instruct_prompt(self, query: str, context: str) -> str:
        """Format prompt for instruct models."""
        tokenizer = self.model_interface.get_tokenizer()
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": f"""Answer the question using ONLY the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer in 1-3 sentences
- Use only information from the context
- If the answer is not in the context, say "Not mentioned in the provided context"
- Be specific and concise

Answer:"""
            }]
            
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback format")
        
        # Fallback format
        return f"""Answer the question using the context below.

Context:
{context}

Question: {query}

Answer:"""
    
    def _format_base_prompt(self, query: str, context: str) -> str:
        """Format prompt for base models."""
        return f"""Context: {context}

Question: {query}

Answer:"""
    
    def _format_instruct_prompt_no_rag(self, query: str) -> str:
        """Format prompt without context for instruct models."""
        tokenizer = self.model_interface.get_tokenizer()
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": f"Question: {query}\n\nAnswer:"
            }]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")
        
        return f"Question: {query}\n\nAnswer:"
    
    def _format_base_prompt_no_rag(self, query: str) -> str:
        """Format prompt without context for base models."""
        return f"Question: {query}\n\nAnswer:"
    
    def _clean_answer(self, answer: str, max_sentences: int = 5) -> str:
        """
        Clean and normalize generated answer.
        
        Args:
            answer: Raw generated text
            max_sentences: Maximum sentences to keep
            
        Returns:
            Cleaned answer
        """
        if not answer or answer.lower() in ['', 'none', 'n/a']:
            return "Not mentioned in the provided context"
        
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove common artifacts
        answer = re.sub(r'^(Answer:|A:)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # Truncate to max sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > max_sentences:
            answer = '. '.join(sentences[:max_sentences])
            # Add period if not present
            if not answer.endswith('.'):
                answer += '.'
        
        return answer