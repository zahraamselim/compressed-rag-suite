"""LLM answer generation with RAG - FIXED for better faithfulness."""

import logging
import re
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Generate answers using LLM with retrieved context.
    FIXED: Better prompts and validation for faithful responses.
    """
    
    def __init__(self, model_interface, config: dict):
        """
        Initialize RAG generator.
        
        Args:
            model_interface: ModelInterface instance
            config: Generation config from config.json
        """
        self.model_interface = model_interface
        
        self.max_new_tokens = config.get('max_new_tokens', 64)
        self.temperature = config.get('temperature', 0.1)  # Slightly raised from 0.0
        self.top_p = config.get('top_p', 0.9)
        self.do_sample = config.get('do_sample', False)
        self.repetition_penalty = config.get('repetition_penalty', 1.2)
        
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
        
        # Generate with slightly relaxed parameters
        answer = self.model_interface.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,  # Use config temperature
            do_sample=self.do_sample,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        
        answer = self._clean_answer(answer)
        
        # Validate and potentially regenerate
        if not self._validate_answer(answer, context, query):
            logger.warning(f"Answer failed validation. Query: {query[:50]}...")
            logger.warning(f"Answer: {answer[:100]}...")
            
            # Try regeneration with stricter prompt
            logger.info("Retrying with stricter prompt...")
            strict_prompt = self._format_strict_prompt(query, context)
            answer = self.model_interface.generate(
                prompt=strict_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,  # Greedy for retry
                do_sample=False,
                repetition_penalty=self.repetition_penalty
            )
            answer = self._clean_answer(answer)
            
            # If still fails, return safe fallback
            if not self._validate_answer(answer, context, query):
                logger.warning("Retry also failed validation. Using fallback.")
                answer = "Based on the provided context, the information is not clearly specified."
        
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
        """
        FIXED: Much stronger prompt that forces context-only answers.
        """
        tokenizer = self.model_interface.get_tokenizer()
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": f"""You are a precise question-answering assistant. You must ONLY use information from the context provided below.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Read the context carefully
- Answer using ONLY information found in the context above
- If the context doesn't contain the answer, respond with: "The context does not provide this information."
- Be concise (1-3 sentences maximum)
- Do NOT use any external knowledge
- Quote or paraphrase directly from the context

ANSWER:"""
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
        return f"""Context: {context}

Question: {query}

Instructions: Answer using ONLY the context above. If not in context, say "Not provided in context."

Answer:"""
    
    def _format_strict_prompt(self, query: str, context: str) -> str:
        """Extra strict prompt for retry attempts."""
        return f"""CONTEXT (this is the ONLY information you can use):
---
{context}
---

QUESTION: {query}

STRICT RULES:
1. You MUST extract your answer DIRECTLY from the context above
2. Do NOT add any information not in the context
3. If the answer is not in the context, respond: "Not provided in context."
4. Maximum 2 sentences

ANSWER (extract from context):"""
    
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
    
    def _clean_answer(self, answer: str, max_sentences: int = 3) -> str:
        """
        Clean and normalize generated answer.
        Limit to max_sentences.
        """
        if not answer or answer.lower() in ['', 'none', 'n/a']:
            return "The information is not provided in the given context."
        
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove common artifacts
        answer = re.sub(r'^(Answer:|A:)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # Split into sentences more carefully
        sentences = []
        current = ""
        
        for char in answer:
            current += char
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
    
    def _validate_answer(
        self, 
        answer: str, 
        context: str, 
        query: str = "",
        min_overlap: float = 0.4,  # Increased from 0.3
        min_context_words: int = 3  # NEW: require minimum context words
    ) -> bool:
        """
        FIXED: Better validation that checks for hallucination.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            query: Original query (to filter out query words)
            min_overlap: Minimum token overlap ratio (0-1)
            min_context_words: Minimum number of context words required in answer
            
        Returns:
            True if answer appears grounded, False if hallucinating
        """
        # Don't validate fallback responses
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in [
            "not provided", "not in the context", "not clearly specified",
            "does not provide", "cannot be answered"
        ]):
            return True
        
        # Tokenize (simple word-based)
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        query_tokens = set(query.lower().split()) if query else set()
        
        # Remove stopwords and short tokens
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'it', 'its', 'as', 'which', 'what',
            'who', 'when', 'where', 'why', 'how', 'can', 'will', 'would', 'should'
        }
        
        # Filter answer tokens
        answer_tokens = {
            t for t in answer_tokens 
            if len(t) > 2 and t not in stopwords and t not in query_tokens
        }
        
        # Filter context tokens
        context_tokens = {
            t for t in context_tokens 
            if len(t) > 2 and t not in stopwords
        }
        
        if not answer_tokens:
            logger.debug("Answer too short for validation")
            return True  # Very short answer, can't validate properly
        
        # Calculate overlap
        overlap_tokens = answer_tokens & context_tokens
        overlap_ratio = len(overlap_tokens) / len(answer_tokens)
        
        # Check minimum context words
        has_min_words = len(overlap_tokens) >= min_context_words
        
        logger.debug(f"Validation: {len(overlap_tokens)}/{len(answer_tokens)} tokens from context ({overlap_ratio:.2%})")
        logger.debug(f"Min words check: {len(overlap_tokens)} >= {min_context_words}: {has_min_words}")
        
        # More lenient: pass if either condition is met
        passed = overlap_ratio >= min_overlap or has_min_words
        
        if not passed:
            logger.warning(f"Validation FAILED:")
            logger.warning(f"  Overlap: {overlap_ratio:.2%} < {min_overlap:.2%}")
            logger.warning(f"  Context words: {len(overlap_tokens)} < {min_context_words}")
            logger.warning(f"  Answer: {answer[:100]}...")
            logger.warning(f"  Missing key words: {answer_tokens - context_tokens}")
        
        return passed