"""LLM answer generation with RAG - Improved version with better faithfulness."""

import logging
import re
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Generate answers using LLM with retrieved context.
    Improved version with strict faithfulness constraints.
    """
    
    def __init__(self, model_interface, config: dict):
        """
        Initialize RAG generator.
        
        Args:
            model_interface: ModelInterface instance
            config: Generation config from config.json
        """
        self.model_interface = model_interface
        
        # Reduced max tokens for more focused answers
        self.max_new_tokens = config.get('max_new_tokens', 64)  # Reduced from 100
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.do_sample = config.get('do_sample', True)
        self.repetition_penalty = config.get('repetition_penalty', 1.2)  # Increased from 1.1
        
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
        
        # Use very strict generation parameters for RAG
        answer = self.model_interface.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,  # Greedy decoding - no randomness
            do_sample=False,  # Deterministic
            top_p=1.0,  # Not used with greedy
            repetition_penalty=self.repetition_penalty
        )
        
        answer = self._clean_answer(answer)
        
        # Validate answer against context
        if not self._validate_answer(answer, context):
            logger.warning("Answer appears to hallucinate - returning fallback")
            answer = "The information is not clearly provided in the given context."
        
        if return_prompt:
            return answer, prompt
        return answer
    
    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """Generate answers for multiple queries in batch."""
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
        """Generate answer without RAG context."""
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
        """Generate answers for multiple queries without context."""
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
        """Format prompt for instruct models with MAXIMUM strictness."""
        tokenizer = self.model_interface.get_tokenizer()
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": f"""Answer the question using ONLY the context below. Do not use any external knowledge.

Context:
{context}

Question: {query}

STRICT RULES:
1. Extract answer DIRECTLY from the context above
2. Use 1-2 sentences maximum
3. If the answer is not in the context, respond: "The information is not provided in the given context."
4. Do NOT add any information that is not explicitly in the context
5. Quote or paraphrase the relevant part of the context

Answer (1-2 sentences):"""
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
        return f"""Answer the question using ONLY the context below.

Context: {context}

Question: {query}

Rules: 
- Answer in 1-2 sentences
- Use ONLY information from the context
- If not in context, say: "The information is not provided in the given context."

Answer:"""
    
    def _format_base_prompt(self, query: str, context: str) -> str:
        """Format prompt for base models."""
        return f"""Context: {context}

Question: {query}

Answer based only on the context above (1-2 sentences):"""
    
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
    
    def _clean_answer(self, answer: str, max_sentences: int = 2) -> str:
        """
        Clean and normalize generated answer.
        Strictly limit to 2 sentences.
        """
        if not answer or answer.lower() in ['', 'none', 'n/a']:
            return "The information is not provided in the given context."
        
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove common artifacts
        answer = re.sub(r'^(Answer:|A:)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # Split into sentences more carefully
        # Handle periods in abbreviations
        sentences = []
        current = ""
        
        for char in answer:
            current += char
            # Sentence ends with . ! ? followed by space or end
            if char in '.!?' and (not current or current[-1] in '.!?'):
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        # Filter out empty sentences
        sentences = [s for s in sentences if len(s.strip()) > 3]
        
        # Take only first N sentences
        if len(sentences) > max_sentences:
            answer = ' '.join(sentences[:max_sentences])
            if not answer.endswith(('.', '!', '?')):
                answer += '.'
        
        return answer
    
    def _validate_answer(self, answer: str, context: str, min_overlap: float = 0.3) -> bool:
        """
        Validate that answer is grounded in context.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            min_overlap: Minimum token overlap ratio (0-1)
            
        Returns:
            True if answer appears grounded, False if hallucinating
        """
        # Don't validate fallback responses
        if "not provided" in answer.lower() or "not in the context" in answer.lower():
            return True
        
        # Tokenize (simple word-based)
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        # Remove stopwords and short tokens
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                     'this', 'that', 'these', 'those', 'it', 'its'}
        
        answer_tokens = {t for t in answer_tokens if len(t) > 2 and t not in stopwords}
        context_tokens = {t for t in context_tokens if len(t) > 2 and t not in stopwords}
        
        if not answer_tokens:
            return True  # Very short answer, can't validate
        
        # Calculate overlap
        overlap = len(answer_tokens & context_tokens)
        overlap_ratio = overlap / len(answer_tokens)
        
        logger.debug(f"Answer validation: {overlap}/{len(answer_tokens)} tokens overlap ({overlap_ratio:.2%})")
        
        return overlap_ratio >= min_overlap
