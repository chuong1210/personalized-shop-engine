import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.schemas import ChatRequest, ChatResponse, AnalysisResult, Entity
from agent.graph import analysis_graph

# Load environment variables
load_dotenv()

# Khởi tạo FastAPI app
app = FastAPI(
    title="Chat Analyzer Agent API",
    description="API phân tích chat e-commerce với LangGraph",
    version="1.0.0"
)

# CORS middleware (cho phép frontend gọi API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Chat Analyzer Agent API is running",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=ChatResponse)
async def analyze_chat(request: ChatRequest):
    """
    Endpoint phân tích chat message
    
    Input: ChatRequest với message
    Output: ChatResponse với AnalysisResult
    """
    try:
        # Chuẩn bị initial state
        initial_state = {
            "user_message": request.message,
            "messages": [],
            "topic": None,
            "sentiment": None,
            "entities": None,
            "purchase_intent": None,
            "confidence": None,
            "reasoning": ""
        }
        
        # Chạy graph
        result = analysis_graph.invoke(initial_state)
        
        # Chuyển entities sang Pydantic model
        entities = [
            Entity(
                type=e["type"],
                value=e["value"],
                context=e.get("context")
            )
            for e in result.get("entities", [])
        ]
        
        # Tạo AnalysisResult
        analysis = AnalysisResult(
            topic=result["topic"],
            sentiment=result["sentiment"],
            entities=entities,
            purchase_intent=result["purchase_intent"],
            confidence=result.get("confidence", 0.8),
            reasoning=result.get("reasoning", "").strip()
        )
        
        return ChatResponse(
            success=True,
            data=analysis,
            error=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Endpoint kiểm tra sức khỏe của service"""
    return {"status": "healthy"}

# Run with: uvicorn src.api.main:app --reload --port 8000