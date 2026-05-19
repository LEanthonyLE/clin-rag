import httpx
from typing import List, Dict, Any, Optional
from agentic_lightrag.config import settings

class LightRAGClient:
    def __init__(self):
        self.base_url = settings.lightrag_api_base_url.rstrip("/")
        self.api_key = settings.lightrag_api_key
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def retrieve(self, query: str, mode: str = "hybrid", top_k: int = 10, workspace: str = None) -> List[Dict[str, Any]]:
        """
        Call LightRAG to retrieve relevant contexts.
        We ask for 'references' and parse them into chunks.
        """
        payload = {
            "query": query,
            "mode": mode,
            "only_need_context": True,
            "top_k": top_k,
            # "include_references": True,
        }
        # Some LightRAG versions support dynamic workspace switching via payload
        if workspace:
            payload["workspace"] = workspace

        print(f"DEBUG: calling LightRAG with payload: {payload}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query",
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                # print(f"DEBUG: LightRAG response keys: {data.keys()}")

                # Priority 1: Check for explicit 'references' list
                if isinstance(data, dict) and data.get("references"):
                    refs = data["references"]
                    results = []
                    for i, ref in enumerate(refs):
                        # Handle case where reference content is null or empty (LightRAG bug/behavior)
                        if isinstance(ref, dict) and not ref.get("content") and not ref.get("text"):
                            continue

                        if isinstance(ref, str):
                            results.append({"text": ref, "score": 1.0, "source": mode, "id": f"ref-{i}"})
                        elif isinstance(ref, dict):
                            # Adapt based on actual fields
                            results.append({
                                "text": ref.get("content") or ref.get("text", ""),
                                "score": ref.get("score", 1.0),
                                "id": ref.get("id", f"ref-{i}"),
                                "source": mode
                            })

                    # Only return if we actually found valid references
                    if results:
                        return results
                    # If we had references but they were all empty/null, fall through to Priority 3

                # Priority 2: Check for 'context' (raw string or list)
                if isinstance(data, dict) and "context" in data:
                    context_content = data["context"]
                    if isinstance(context_content, str):
                         return [{"text": context_content, "score": 1.0, "source": mode}]
                    elif isinstance(context_content, list):
                        return context_content

                # Priority 3: Fallback to 'response' if it looks like context
                if isinstance(data, dict) and "response" in data:
                     # If we asked for only_need_context=True, response IS the context
                     return [{"text": data["response"], "score": 1.0, "source": mode}]

                print("DEBUG: No context found in response")
                return []

            except Exception as e:
                print(f"Error calling LightRAG: {e}")
                return []

# Global instance
lightrag_client = LightRAGClient()
