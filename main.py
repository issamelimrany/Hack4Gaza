import uuid
from typing import List, Dict, Set
from dotenv import load_dotenv
import os
# Remove Supabase import
import psycopg2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from services.chroma_service import ChromaService
from services.llm_service import LLMService
import json

# Load environment variables
load_dotenv()

# Add DB connection helper
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_PORT = os.getenv("port")
DB_NAME = os.getenv("dbname")

def get_db_connection():
    return psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME
    )

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

# In-memory storage for queries and expert responses
queries_db = {}

from datetime import datetime

class UserQuery(BaseModel):
    id: str
    question: str
    assigned_experts: List[ExpertResponse]
    llm_answer: str
    expert_responses: List[dict]  # {expert_id, expert_name, response, timestamp}

class SubmitExpertResponseRequest(BaseModel):
    expert_id: str
    expert_name: str
    response: str

# WebSocket manager for real-time expert response updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # query_id -> set of websockets

    async def connect(self, query_id: str, websocket: WebSocket):
        await websocket.accept()
        if query_id not in self.active_connections:
            self.active_connections[query_id] = set()
        self.active_connections[query_id].add(websocket)

    def disconnect(self, query_id: str, websocket: WebSocket):
        if query_id in self.active_connections:
            self.active_connections[query_id].discard(websocket)
            if not self.active_connections[query_id]:
                del self.active_connections[query_id]

    async def broadcast(self, query_id: str, message: dict):
        if query_id in self.active_connections:
            for connection in list(self.active_connections[query_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.disconnect(query_id, connection)

manager = ConnectionManager()

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
        print(top_experts)
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
        # Store query in DB
        query_id = str(uuid.uuid4())
        queries_db[query_id] = UserQuery(
            id=query_id,
            question=request.query,
            assigned_experts=expert_responses,
            llm_answer=llm_answer,
            expert_responses=[]
        )
        # Store in Postgres
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO queries (id, question, llm_answer, assigned_experts)
                VALUES (%s, %s, %s, %s)
                """,
                (query_id, request.query, llm_answer, json.dumps([ex.dict() for ex in expert_responses]))
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_e:
            raise HTTPException(status_code=500, detail=f"Postgres error: {db_e}")
        return {"experts": expert_responses, "llm_answer": llm_answer, "query_id": query_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/query/{query_id}")
async def get_query(query_id: str):
    query = queries_db.get(query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    return query

@app.get("/query_list")
async def get_query_list():
    # Return all queries in the queries_db
    return {
        "queries": [
            {
                "id": q.id,
                "question": q.question,
                "assigned_experts": [
                    {
                        "id": ex.id,
                        "name": ex.name,
                        "expertise": ex.expertise,
                        "description": ex.description,
                        "similarity_score": ex.similarity_score,
                    } for ex in q.assigned_experts
                ],
            }
            for q in queries_db.values()
        ]
    }

@app.websocket("/ws/query/{query_id}")
async def websocket_endpoint(websocket: WebSocket, query_id: str):
    await manager.connect(query_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive, ignore input
    except WebSocketDisconnect:
        manager.disconnect(query_id, websocket)

@app.post("/query/{query_id}/expert_response")
async def submit_expert_response(query_id: str, req: SubmitExpertResponseRequest):
    query = queries_db.get(query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    # Add expert response
    response_obj = {
        "expert_id": req.expert_id,
        "expert_name": req.expert_name,
        "response": req.response,
        "timestamp": datetime.utcnow().isoformat()
    }
    query.expert_responses.append(response_obj)
    # Store in Postgres
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO answers (query_id, expert_id, expert_name, response, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (query_id, req.expert_id, req.expert_name, req.response, response_obj["timestamp"])
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as db_e:
        raise HTTPException(status_code=500, detail=f"Postgres error: {db_e}")
    # Broadcast to all websocket clients listening for this query
    await manager.broadcast(query_id, {"type": "expert_response", "data": response_obj})
    return {"detail": "Expert response submitted."}

@app.post("/experts", response_model=AddExpertResponse)
async def add_expert(request: AddExpertRequest):
    """Add a new expert to the ChromaDB collection and Postgres"""
    try:
        expert = {
            "id": str(uuid.uuid4()),
            "name": request.name,
            "expertise": request.expertise,
            "description": request.description
        }
        chroma_service.add_expert(expert)
        # Store in Postgres
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experts (id, name, expertise, description)
                VALUES (%s, %s, %s, %s)
                """,
                (expert["id"], expert["name"], expert["expertise"], expert["description"])
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_e:
            raise HTTPException(status_code=500, detail=f"Postgres error: {db_e}")
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

@app.delete("/queries")
async def clear_all_queries():
    queries_db.clear()
    return {"detail": "All queries and answers deleted successfully."}

@app.get("/all_answers")
async def get_all_answers():
    # Return all queries with their expert responses
    return {
        "queries": [
            {
                "id": q.id,
                "question": q.question,
                "assigned_experts": [
                    {
                        "id": ex.id,
                        "name": ex.name,
                        "expertise": ex.expertise,
                        "description": ex.description,
                        "similarity_score": ex.similarity_score,
                    } for ex in q.assigned_experts
                ],
                "expert_responses": q.expert_responses,
            }
            for q in queries_db.values()
        ]
    }


