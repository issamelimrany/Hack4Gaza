import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import re

class SimilaritySearch:
    def __init__(self):
        try:
            # Initialize the sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load sentence transformer model: {e}")
            self.model_loaded = False
    
    def find_similar_experts(self, query: str, experts: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the top-k most similar experts to the given query using semantic similarity
        """
        if not experts:
            return []
        
        if not self.model_loaded:
            return self._fallback_similarity_search(query, experts, top_k)
        
        try:
            # Prepare expert texts for embedding
            expert_texts = []
            for expert in experts:
                # Combine expertise and description for better matching
                expert_text = f"{expert.get('expertise', '')} {expert.get('description', '')}"
                expert_texts.append(expert_text)
            
            # Generate embeddings
            query_embedding = self.model.encode([query])
            expert_embeddings = self.model.encode(expert_texts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, expert_embeddings)[0]
            
            # Add similarity scores to experts and sort
            for i, expert in enumerate(experts):
                expert['similarity_score'] = float(similarities[i])
            
            # Sort by similarity score (descending) and return top-k
            sorted_experts = sorted(experts, key=lambda x: x['similarity_score'], reverse=True)
            return sorted_experts[:top_k]
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return self._fallback_similarity_search(query, experts, top_k)
    
    def fallback_similarity_search(self, query: str, experts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback similarity search using simple keyword matching when sentence transformers is not available
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        scored_experts = []
        
        for expert in experts:
            # Combine expertise and description
            expert_text = f"{expert.get('expertise', '')} {expert.get('description', '')}".lower()
            expert_words = set(re.findall(r'\w+', expert_text))
            
            # Calculate simple word overlap score
            if expert_words:
                overlap = len(query_words.intersection(expert_words))
                total_words = len(query_words.union(expert_words))
                similarity_score = overlap / total_words if total_words > 0 else 0
            else:
                similarity_score = 0
            
            expert_copy = expert.copy()
            expert_copy['similarity_score'] = similarity_score
            scored_experts.append(expert_copy)
        
        # Sort by similarity score and return top-k
        sorted_experts = sorted(scored_experts, key=lambda x: x['similarity_score'], reverse=True)
        return sorted_experts[:top_k]