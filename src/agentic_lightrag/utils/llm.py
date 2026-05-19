import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from agentic_lightrag.config import settings

T = TypeVar("T", bound=BaseModel)

class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.llm_model

    async def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """
        Generate simple text response
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content

    async def generate_structured(self, system_prompt: str, user_prompt: str, response_model: Type[T], temperature: float = 0.1) -> T:
        """
        Generate structured JSON response matching a Pydantic model.
        Uses JSON mode if available, or instruction following.
        """
        # Force JSON instruction
        json_instruction = f"\n\nRespond strictly with a valid JSON object matching this schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}"

        full_system_prompt = system_prompt + json_instruction

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        try:
            # Parse JSON and validate with Pydantic
            data = json.loads(content)
            return response_model.model_validate(data)
        except Exception as e:
            # Simple retry or fallback could go here
            raise ValueError(f"Failed to parse LLM JSON output: {content} | Error: {e}")

# Global instance
llm_client = LLMClient()
