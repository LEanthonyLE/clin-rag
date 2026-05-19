from typing import Dict, Any
from agentic_lightrag.skills.implementations import (
    QueryRewriteSkill,
    HybridRetrieveSkill,
    AnswerWithCitationsSkill
)
# Keep Rerank as stub for now (Phase 3)
from agentic_lightrag.skills.stubs import RerankSkill

from agentic_lightrag.schemas.contracts import (
    QueryRewriteInput,
    HybridRetrieveInput,
    RerankInput,
    AnswerInput
)

class SimpleOrchestrator:
    """
    Phase 1 Orchestrator: Linear pipeline execution.
    Query -> Rewrite -> Retrieve -> Rerank -> Answer
    """
    def __init__(self):
        # Initialize skills (stateless in Phase 1)
        self.rewrite_skill = QueryRewriteSkill()
        self.retrieve_skill = HybridRetrieveSkill()
        self.rerank_skill = RerankSkill()
        self.answer_skill = AnswerWithCitationsSkill()

    async def run_pipeline(self, query: str, workspace: str = "default") -> Dict[str, Any]:
        traces = []

        # 1. Query Rewrite
        rewrite_res = await self.rewrite_skill.execute(QueryRewriteInput(
            query=query,
            workspace=workspace
        ))
        traces.append(rewrite_res.trace)
        if not rewrite_res.ok:
            return {"error": "Rewrite failed", "traces": traces}

        rewritten_queries = rewrite_res.data['rewritten_queries']
        # Ensure original query is included
        all_queries = [query] + rewritten_queries

        # 2. Hybrid Retrieve (Vector + BM25)
        retrieve_res = await self.retrieve_skill.execute(HybridRetrieveInput(
            queries=all_queries,
            workspace=workspace,
            top_k=20 # Get more for reranking
        ))
        traces.append(retrieve_res.trace)
        if not retrieve_res.ok:
            return {"error": "Retrieval failed", "traces": traces}

        raw_evidences = retrieve_res.data['evidences']

        # 3. Rerank
        # Convert dict back to Evidence objects for the next skill
        # (In a real app, Pydantic handles this, but here we are passing between skills)
        from agentic_lightrag.schemas.common import Evidence
        evidences_objs = [Evidence(**e) for e in raw_evidences]

        rerank_res = await self.rerank_skill.execute(RerankInput(
            query=query,
            evidences=evidences_objs,
            top_k=5
        ))
        traces.append(rerank_res.trace)

        top_evidences_data = rerank_res.data['evidences']
        top_evidences = [Evidence(**e) for e in top_evidences_data]

        # 4. Answer Generation
        answer_res = await self.answer_skill.execute(AnswerInput(
            query=query,
            evidences=top_evidences,
            workspace=workspace
        ))
        traces.append(answer_res.trace)

        return {
            "answer": answer_res.data['answer'],
            "citations": answer_res.data['citations'],
            "gaps": answer_res.data.get('gaps', []),
            "traces": [t.model_dump() for t in traces] # Return full trace history
        }
