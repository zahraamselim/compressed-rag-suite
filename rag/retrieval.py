"""Context retrieval from vector store."""

from typing import List, Dict, Optional
import numpy as np
import logging

from rag.indexing import VectorStore
from rag.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Retrieve relevant context for queries.
    Includes re-ranking and diversity mechanisms.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        config: dict
    ):
        """
        Initialize context retriever.
        
        Args:
            vector_store: VectorStore instance
            embedding_model: EmbeddingModel instance
            config: Retrieval config from config.json
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
        self.top_k = config.get('top_k', 3)
        self.similarity_threshold = config.get('similarity_threshold', 0.0)
        self.rerank = config.get('rerank', False)
        self.diversity_penalty = config.get('diversity_penalty', 0.0)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve (overrides config)
            filters: Metadata filters
            
        Returns:
            List of dicts with 'text', 'score', 'metadata', 'chunk_id'
        """
        k = top_k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k * 2 if self.rerank else k,  # Get more if reranking
                where=filters
            )
            
            # Check if we got results
            if not results['ids'][0]:
                logger.warning("No results found for query")
                return []
            
            # Format results
            retrieved_chunks = []
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity score
                # ChromaDB with cosine space returns L2 distance in range [0, 2]
                # For cosine similarity: similarity = 1 - (distance / 2)
                distance = results['distances'][0][i]
                similarity_score = 1.0 - (distance / 2.0)  # More explicit conversion
                
                chunk_data = {
                    'text': results['documents'][0][i],
                    'score': max(0.0, min(1.0, similarity_score)),  # Clamp to [0, 1]
                    'distance': distance,  # Keep original distance for reference
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'chunk_id': results['ids'][0][i]
                }
                
                # Apply similarity threshold
                if chunk_data['score'] >= self.similarity_threshold:
                    retrieved_chunks.append(chunk_data)
            
            if not retrieved_chunks:
                logger.warning(f"No chunks passed similarity threshold of {self.similarity_threshold}")
                return []
            
            # Re-rank if enabled
            if self.rerank and len(retrieved_chunks) > k:
                retrieved_chunks = self._rerank(query, retrieved_chunks, k)
            else:
                retrieved_chunks = retrieved_chunks[:k]
            
            # Apply diversity penalty if enabled
            if self.diversity_penalty > 0 and len(retrieved_chunks) > 1:
                retrieved_chunks = self._apply_diversity(retrieved_chunks)
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def get_context_string(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n"
    ) -> str:
        """
        Retrieve and format context as a single string.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            separator: Separator between chunks
            
        Returns:
            Formatted context string
        """
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return ""
        
        context_parts = [chunk['text'] for chunk in chunks]
        return separator.join(context_parts)
    
    def _rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """
        Simple re-ranking based on token overlap.
        More sophisticated methods can use cross-encoders.
        
        Args:
            query: Query string
            chunks: List of chunk dicts
            top_k: Number of top chunks to return
            
        Returns:
            Re-ranked chunks
        """
        query_tokens = set(query.lower().split())
        
        for chunk in chunks:
            chunk_tokens = set(chunk['text'].lower().split())
            overlap = len(query_tokens & chunk_tokens)
            
            # Combine with original score
            overlap_score = overlap / max(len(query_tokens), 1)
            chunk['rerank_score'] = chunk['score'] * 0.7 + overlap_score * 0.3
        
        # Sort by rerank score
        chunks.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)
        return chunks[:top_k]
    
    def _apply_diversity(self, chunks: List[Dict]) -> List[Dict]:
        """
        Apply maximal marginal relevance (MMR) for diversity.
        
        Args:
            chunks: List of chunk dicts
            
        Returns:
            Diversified chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        # Simple diversity: penalize chunks that are too similar to already selected ones
        selected = [chunks[0]]
        remaining = chunks[1:].copy()
        
        while len(selected) < len(chunks) and remaining:
            best_chunk = None
            best_score = -float('inf')
            
            for chunk in remaining:
                # Calculate similarity to selected chunks
                chunk_text_set = set(chunk['text'].lower().split())
                max_sim = 0.0
                
                for sel_chunk in selected:
                    sel_text_set = set(sel_chunk['text'].lower().split())
                    union_size = len(chunk_text_set | sel_text_set)
                    if union_size > 0:
                        overlap = len(chunk_text_set & sel_text_set)
                        sim = overlap / union_size
                        max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = chunk['score'] - self.diversity_penalty * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = chunk
            
            if best_chunk:
                selected.append(best_chunk)
                remaining.remove(best_chunk)
            else:
                break
        
        return selected
