"""LLM answer generation with RAG."""

import logging
import re

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
    ) -> str:
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
                logger.warning(f"Chat template failed: {e}")
        
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
            except:
                pass
        
        return f"Question: {query}\n\nAnswer:"
    
    def _format_base_prompt_no_rag(self, query: str) -> str:
        """Format prompt without context for base models."""
        return f"Question: {query}\n\nAnswer:"
    
    def _clean_answer(self, answer: str, max_sentences: int = 5) -> str:
        """Clean and normalize generated answer."""
        if not answer or answer.lower() in ['', 'none', 'n/a']:
            return "Not mentioned in the provided context"
        
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Truncate to max sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > max_sentences:
            answer = '. '.join(sentences[:max_sentences]) + '.'
        
        return answer
