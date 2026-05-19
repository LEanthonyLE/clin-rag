from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# --- 基础组件 ---

class TraceInfo(BaseModel):
    """
    全链路追踪信息，用于回放和调试
    """
    skill: str = Field(..., description="调用的 Skill 名称")
    start_ts: datetime = Field(default_factory=datetime.now)
    end_ts: Optional[datetime] = None
    latency_ms: Optional[float] = None
    inputs_digest: Optional[str] = None # 输入摘要，用于去重或缓存键
    counts: Dict[str, int] = Field(default_factory=dict) # 统计信息，如 token 数、检索条数
    errors: List[str] = Field(default_factory=list)

class Evidence(BaseModel):
    """
    核心证据对象，贯穿 Vector/BM25/KG 所有环节
    """
    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:8]}")
    chunk_id: str = Field(..., description="原始 LightRAG chunk ID")
    doc_id: Optional[str] = None
    page_range: Optional[str] = None
    chapter_path: Optional[str] = None
    text: str = Field(..., description="证据原文片段")
    score: float = Field(..., description="归一化分数 0-1")
    source: List[Literal["vector", "bm25", "kg_path"]] = Field(..., description="证据来源")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据：topic, department等")

# --- 统一响应结构 ---

class SkillResponse(BaseModel):
    """
    所有 Skill 的统一返回格式
    """
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    workspace: str = "default"
    ok: bool = True
    data: Dict[str, Any] = Field(default_factory=dict, description="Skill 的具体输出")
    trace: TraceInfo

# --- 具体 Skill 的输入输出 Payload 定义 (示例) ---

class QueryRewriteInput(BaseModel):
    query: str
    workspace: str = "default"
    top_k: int = 3

class HybridRetrieveInput(BaseModel):
    rewritten_queries: List[str]
    metadata_filters: Dict[str, Any] = {}
    top_k: int = 10
    workspace: str = "default"
