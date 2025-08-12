import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai

class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        genai.configure(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings

class ChromaService:
    def __init__(self):
        # Initialize ChromaDB with persistent storage
        self.db_path = os.path.join(os.getcwd(), "data", "chroma_db")
        os.makedirs(self.db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use the Gemini embedding function
        self.embedding_function = GeminiEmbeddingFunction()
        # Get or create the experts collection with the custom embedding function
        self.collection = self.client.get_or_create_collection(
            name="experts",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_expert(self, expert: Dict[str, Any]):
        """Add a new expert to the collection"""
        
        # Combine expertise and description for embedding
        text = f"{expert['expertise']} {expert['description']}"
        
        self.collection.add(
            documents=[text],
            metadatas=[{
                "name": expert["name"],
                "expertise": expert["expertise"],
                "description": expert["description"]
            }],
            ids=[expert["id"]]
        )
    
    async def get_experts(self) -> List[Dict[str, Any]]:
        """Get all experts from the collection"""
        try:
            results = self.collection.get()
            
            experts = []
            for i, doc_id in enumerate(results["ids"]):
                expert = {
                    "id": doc_id,
                    "name": results["metadatas"][i]["name"],
                    "expertise": results["metadatas"][i]["expertise"],
                    "description": results["metadatas"][i]["description"]
                }
                experts.append(expert)
            
            return experts
            
        except Exception as e:
            print(f"Error fetching experts from ChromaDB: {e}")
            return []
    
    async def search_similar_experts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experts using ChromaDB's built-in similarity search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            experts = []
            for i, doc_id in enumerate(results["ids"][0]):
                expert = {
                    "id": doc_id,
                    "name": results["metadatas"][0][i]["name"],
                    "expertise": results["metadatas"][0][i]["expertise"],
                    "description": results["metadatas"][0][i]["description"],
                    "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                experts.append(expert)
            
            return experts
            
        except Exception as e:
            print(f"Error searching experts in ChromaDB: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name,
            "metadata": self.collection.metadata
        } 