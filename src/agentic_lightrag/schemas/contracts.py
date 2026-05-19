from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from agentic_lightrag.schemas.common import Evidence

# --- Query Rewrite ---
class QueryRewriteInput(BaseModel):
    query: str
    workspace: str = "default"
    n_rewrites: int = 3

class QueryRewriteOutput(BaseModel):
    original_query: str
    rewritten_queries: List[str]

# --- Hybrid Retrieve ---
class HybridRetrieveInput(BaseModel):
    queries: List[str] = Field(..., description="Original query + rewrites")
    workspace: str = "default"
    top_k: int = 10
    filters: Dict[str, Any] = Field(default_factory=dict)

class HybridRetrieveOutput(BaseModel):
    evidences: List[Evidence]

# --- Rerank ---
class RerankInput(BaseModel):
    query: str
    evidences: List[Evidence]
    top_k: int = 5

class RerankOutput(BaseModel):
    evidences: List[Evidence]

# --- Answer ---
class AnswerInput(BaseModel):
    query: str
    evidences: List[Evidence]
    workspace: str = "default"

class AnswerOutput(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] # Simplified citation object
    gaps: List[str] = [] # Missing information
