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
        # Override temperature for RAG to reduce hallucination
        answer = self.model_interface.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,  # Very low for factual extraction
            do_sample=False,  # Greedy decoding for consistency
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
        """Format prompt for instruct models with strict constraints."""
        tokenizer = self.model_interface.get_tokenizer()
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": f"""You are a precise information extraction assistant. Answer the question using ONLY information from the provided context. Do not add any information that is not explicitly stated in the context.

Context:
{context}

Question: {query}

CRITICAL INSTRUCTIONS:
- Extract the answer directly from the context above
- Use 1-3 complete sentences maximum
- Do NOT invent, infer, or add any information not in the context
- If the exact answer is not in the context, respond ONLY with: "The information is not provided in the given context."
- Quote or closely paraphrase the relevant text
- Be factual and concise

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
        
        # Fallback format with strict constraints
        return f"""You must answer using ONLY the information provided in the context below. Do not add any external knowledge.

Context:
{context}

Question: {query}

Instructions:
- Answer in 1-3 sentences
- Use ONLY information from the context
- If not in context, say: "The information is not provided in the given context."

Answer:"""
    
    def _format_base_prompt(self, query: str, context: str) -> str:
        """Format prompt for base models."""
        return f"""Use the context below to answer the question. Only use information from the context.

Context: {context}

Question: {query}

Answer based only on the context above:"""
    
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
    
    def _clean_answer(self, answer: str, max_sentences: int = 3) -> str:
        """
        Clean and normalize generated answer.
        
        Args:
            answer: Raw generated text
            max_sentences: Maximum sentences to keep (reduced to 3)
            
        Returns:
            Cleaned answer
        """
        if not answer or answer.lower() in ['', 'none', 'n/a']:
            return "The information is not provided in the given context."
        
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove common artifacts
        answer = re.sub(r'^(Answer:|A:)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # Truncate to max sentences (now 3 instead of 5)
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > max_sentences:
            answer = '. '.join(sentences[:max_sentences])
            # Add period if not present
            if not answer.endswith('.'):
                answer += '.'
        
        return answer