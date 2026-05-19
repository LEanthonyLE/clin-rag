from typing import List
from agentic_lightrag.skills.base import BaseSkill
from agentic_lightrag.schemas.contracts import (
    QueryRewriteInput, QueryRewriteOutput,
    HybridRetrieveInput, HybridRetrieveOutput,
    RerankInput, RerankOutput,
    AnswerInput, AnswerOutput
)
from agentic_lightrag.schemas.common import Evidence

class QueryRewriteSkill(BaseSkill[QueryRewriteInput, QueryRewriteOutput]):
    name = "QueryRewriteSkill"

    async def _execute_impl(self, params: QueryRewriteInput) -> QueryRewriteOutput:
        # Mock implementation for Phase 0
        return QueryRewriteOutput(
            original_query=params.query,
            rewritten_queries=[
                f"{params.query} (medical term expansion)",
                f"{params.query} (synonym 1)",
                f"{params.query} (related concept)"
            ][:params.n_rewrites]
        )

class HybridRetrieveSkill(BaseSkill[HybridRetrieveInput, HybridRetrieveOutput]):
    name = "HybridRetrieveSkill"

    async def _execute_impl(self, params: HybridRetrieveInput) -> HybridRetrieveOutput:
        # Mock evidence for Phase 0
        mock_evidences = []
        for i, q in enumerate(params.queries):
            mock_evidences.append(Evidence(
                chunk_id=f"chunk-{i}",
                text=f"Mock evidence content for query: {q}",
                score=0.95 - (i * 0.05),
                source=["vector"],
                metadata={"topic": "medical"}
            ))

        return HybridRetrieveOutput(
            evidences=mock_evidences[:params.top_k]
        )

class RerankSkill(BaseSkill[RerankInput, RerankOutput]):
    name = "RerankSkill"

    async def _execute_impl(self, params: RerankInput) -> RerankOutput:
        # Mock rerank (just return top k sorted by score)
        sorted_evidences = sorted(params.evidences, key=lambda x: x.score, reverse=True)
        return RerankOutput(
            evidences=sorted_evidences[:params.top_k]
        )

class AnswerWithCitationsSkill(BaseSkill[AnswerInput, AnswerOutput]):
    name = "AnswerWithCitationsSkill"

    async def _execute_impl(self, params: AnswerInput) -> AnswerOutput:
        # Mock answer generation
        if not params.evidences:
            return AnswerOutput(
                answer="I could not find enough evidence to answer your question.",
                citations=[],
                gaps=["Missing clinical trials data"]
            )

        return AnswerOutput(
            answer=f"Based on the retrieved evidence, here is the answer to '{params.query}'. The system found {len(params.evidences)} relevant chunks.",
            citations=[
                {"evidence_id": ev.evidence_id, "text_snippet": ev.text[:50]}
                for ev in params.evidences
            ]
        )
