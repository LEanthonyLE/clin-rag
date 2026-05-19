from typing import List, Dict, Any
from agentic_lightrag.skills.base import BaseSkill
from agentic_lightrag.schemas.contracts import (
    QueryRewriteInput, QueryRewriteOutput,
    HybridRetrieveInput, HybridRetrieveOutput,
    AnswerInput, AnswerOutput
)
from agentic_lightrag.schemas.common import Evidence
from agentic_lightrag.utils.llm import llm_client
from agentic_lightrag.utils.lightrag_client import lightrag_client
from pydantic import BaseModel, Field
import asyncio

# --- Real Query Rewrite using LLM ---

class RewriteResponseModel(BaseModel):
    rewritten_queries: List[str]

class QueryRewriteSkill(BaseSkill[QueryRewriteInput, QueryRewriteOutput]):
    name = "QueryRewriteSkill"

    async def _execute_impl(self, params: QueryRewriteInput) -> QueryRewriteOutput:
        print(f"DEBUG: QueryRewriteSkill processing query: '{params.query}'")
        # 如果没有配置 LLM，回退到简单的字符串操作
        if not llm_client.client.api_key:
            print("DEBUG: No LLM API Key, skipping rewrite.")
            return QueryRewriteOutput(
                original_query=params.query,
                rewritten_queries=[params.query]
            )

        system_prompt = """
        You are a helpful assistant that rewrites user queries for a RAG system.
        Generate 3-5 variations of the user's query to improve retrieval coverage.
        Include synonyms, medical terminology expansion if applicable, and keyword-focused queries.
        """

        user_prompt = f"Original Query: {params.query}"

        try:
            result = await llm_client.generate_structured(
                system_prompt,
                user_prompt,
                RewriteResponseModel
            )
            rewrites = result.rewritten_queries[:params.n_rewrites]
            print(f"DEBUG: QueryRewriteSkill generated rewrites: {rewrites}")
        except Exception as e:
            print(f"DEBUG: LLM Rewrite failed: {e}")
            rewrites = [params.query]

        return QueryRewriteOutput(
            original_query=params.query,
            rewritten_queries=rewrites
        )


# --- Real Retrieval using LightRAG API ---

class HybridRetrieveSkill(BaseSkill[HybridRetrieveInput, HybridRetrieveOutput]):
    name = "HybridRetrieveSkill"

    async def _execute_impl(self, params: HybridRetrieveInput) -> HybridRetrieveOutput:
        # 收集所有查询 (原始 + 扩写)
        unique_queries = list(set(params.queries))
        print(f"DEBUG: HybridRetrieveSkill executing for queries: {unique_queries}")

        all_evidences = []

        try:
            tasks = []
            for q in unique_queries:
                # 调用 LightRAG
                # User requested to turn off hybrid retrieval for debugging.
                # Trying 'naive' mode which retrieves chunks by vector similarity directly.
                tasks.append(lightrag_client.retrieve(q, mode="naive", workspace=params.workspace))

            results_list = await asyncio.gather(*tasks)

            # 展平并转换为 Evidence 对象
            for i, results in enumerate(results_list):
                query_source = unique_queries[i]
                print(f"DEBUG: Retrieval for '{query_source}' returned {len(results)} items")
                for item in results:
                    # 假设 item 是 {'text': '...', 'score': ...}
                    text = item.get("text", "")
                    if not text:
                        continue

                    all_evidences.append(Evidence(
                        chunk_id=item.get("id", f"temp-{len(all_evidences)}"),
                        text=text,
                        score=item.get("score", 0.8),
                        source=["vector"], # 暂时假定为 vector，后续根据 LightRAG 返回细分
                        metadata={"query_source": query_source}
                    ))

        except Exception as e:
            print(f"DEBUG: Retrieval failed exception: {e}")
            print(f"Retrieval failed: {e}")

        # 简单的去重 (按 text 内容)
        seen = set()
        unique_evidences = []
        for ev in all_evidences:
            if ev.text not in seen:
                seen.add(ev.text)
                unique_evidences.append(ev)

        print(f"DEBUG: HybridRetrieveSkill total unique evidences found: {len(unique_evidences)}")
        return HybridRetrieveOutput(
            evidences=unique_evidences[:params.top_k]
        )


# --- Real Answer Generation using LLM ---

class AnswerWithCitationsSkill(BaseSkill[AnswerInput, AnswerOutput]):
    name = "AnswerWithCitationsSkill"

    async def _execute_impl(self, params: AnswerInput) -> AnswerOutput:
        print(f"DEBUG: AnswerWithCitationsSkill received {len(params.evidences)} evidences for answering")
        if not params.evidences:
             print("DEBUG: No evidences provided to AnswerSkill")
             return AnswerOutput(
                answer="I could not find any relevant information in the knowledge base.",
                citations=[],
                gaps=["No evidence found"]
            )

        # 构建 Prompt
        context_str = ""
        for i, ev in enumerate(params.evidences):
            context_str += f"[{i+1}] {ev.text}\n\n"

        system_prompt = """
        You are an expert assistant answering questions based STRICTLY on the provided context.
        1. Use the provided context to answer the user's question.
        2. If the context doesn't contain the answer, admit it.
        3. Cite your sources using [1], [2] notation corresponding to the context chunks.
        """

        user_prompt = f"Question: {params.query}\n\nContext:\n{context_str}"

        try:
            answer_text = await llm_client.generate_text(
                system_prompt,
                user_prompt
            )

            # 简单的引用提取 (这里仅做示例，实际上可以让 LLM 输出结构化引用)
            citations = []
            for i, ev in enumerate(params.evidences):
                ref_marker = f"[{i+1}]"
                if ref_marker in answer_text:
                    citations.append({
                        "evidence_id": ev.evidence_id,
                        "ref_index": i+1,
                        "text_snippet": ev.text[:50] + "..."
                    })

            return AnswerOutput(
                answer=answer_text,
                citations=citations
            )

        except Exception as e:
            return AnswerOutput(
                answer=f"Error generating answer: {str(e)}",
                citations=[],
                gaps=["LLM generation failed"]
            )
