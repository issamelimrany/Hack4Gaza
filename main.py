import uuid
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from services.chroma_service import ChromaService
from services.llm_service import LLMService

# Load environment variables
load_dotenv()

app = FastAPI(title="Hack4Gaza", description="Hack4Gaza")

# Initialize services
chroma_service = ChromaService()
llm_service = LLMService()

class QueryRequest(BaseModel):
    query: str

class ExpertResponse(BaseModel):
    id: str
    name: str
    expertise: str
    description: str
    similarity_score: float

class QueryResponse(BaseModel):
    experts: List[ExpertResponse]
    llm_answer: str

class AddExpertRequest(BaseModel):
    name: str = Field(..., example="Jane Doe")
    expertise: str = Field(..., example="Machine Learning, AI")
    description: str = Field(..., example="Expert in ML and AI with 10 years of experience.")

class AddExpertResponse(BaseModel):
    id: str
    name: str
    expertise: str
    description: str

@app.get("/")
async def root():
    return {"message": "All is up nd running"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Use ChromaDB's built-in similarity search to get top 5 experts
        top_experts = await chroma_service.search_similar_experts(
            query=request.query,
            top_k=5
        )
        
        if not top_experts:
            raise HTTPException(status_code=404, detail="No experts found in database")
        
        # Get LLM answer for the query
        llm_answer = await llm_service.get_answer(request.query)
        
        # Format response
        expert_responses = [
            ExpertResponse(
                id=expert["id"],
                name=expert["name"],
                expertise=expert["expertise"],
                description=expert["description"],
                similarity_score=expert["similarity_score"]
            )
            for expert in top_experts
        ]
        
        return QueryResponse(
            experts=expert_responses,
            llm_answer=llm_answer
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/experts", response_model=AddExpertResponse)
async def add_expert(request: AddExpertRequest):
    """Add a new expert to the ChromaDB collection"""
    try:
        expert = {
            "id": str(uuid.uuid4()),
            "name": request.name,
            "expertise": request.expertise,
            "description": request.description
        }
        chroma_service.add_expert(expert)
        return AddExpertResponse(**expert)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding expert: {str(e)}")



@app.get("/experts/info")
async def get_experts_info():
    """Get information about the experts collection"""
    try:
        info = chroma_service.get_collection_info()
        return {
            "collection_name": info["name"],
            "expert_count": info["count"],
            "metadata": info["metadata"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")

@app.get("/experts")
async def get_all_experts():
    """Get all experts from the collection"""
    try:
        experts = await chroma_service.get_experts()
        return {"experts": experts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting experts: {str(e)}")
    


@app.delete("/experts")
async def clear_all_experts():
    """Delete all experts from the ChromaDB collection"""
    try:
        # Get all IDs in the collection
        results = chroma_service.collection.get()
        all_ids = results["ids"]
        if all_ids:
            chroma_service.collection.delete(ids=all_ids)
        return {"detail": "All experts deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing experts: {str(e)}")


