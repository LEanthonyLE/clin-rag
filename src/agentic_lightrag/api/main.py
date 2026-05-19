from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from agentic_lightrag.orchestrator.simple_pipeline import SimpleOrchestrator

app = FastAPI(
    title="Agentic LightRAG Orchestrator",
    description="Orchestrator for LightRAG with Agentic capabilities (Phase 1)",
    version="0.1.0"
)

# Initialize Orchestrator
orchestrator = SimpleOrchestrator()

class ChatRequest(BaseModel):
    query: str
    workspace: str = "default"
    history: Optional[List[Dict[str, str]]] = None # For future multi-turn

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    gaps: List[str]
    traces: List[Dict[str, Any]]

@app.get("/health")
async def health_check():
    return {"status": "ok", "phase": "Phase 1: Text End-to-End"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = await orchestrator.run_pipeline(
            query=request.query,
            workspace=request.workspace
        )
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
