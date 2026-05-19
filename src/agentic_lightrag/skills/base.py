from abc import ABC, abstractmethod
from datetime import datetime
import time
from typing import Any, Dict, Type, Generic, TypeVar
from pydantic import BaseModel

from agentic_lightrag.schemas.common import SkillResponse, TraceInfo

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

class BaseSkill(ABC, Generic[InputT, OutputT]):
    """
    Abstract Base Class for all Skills.
    Enforces the unified response structure and automatic tracing.
    """
    name: str = "BaseSkill"

    async def execute(self, params: InputT) -> SkillResponse:
        start_ts = datetime.now()
        start_time = time.time()

        trace = TraceInfo(
            skill=self.name,
            start_ts=start_ts,
            inputs_digest=str(params.model_dump())[:100] + "..." # Simple digest
        )

        try:
            # Execute the concrete logic
            result_data = await self._execute_impl(params)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            trace.end_ts = datetime.now()
            trace.latency_ms = latency_ms
            trace.counts["input_keys"] = len(params.model_dump())

            return SkillResponse(
                workspace=getattr(params, "workspace", "default"),
                ok=True,
                data=result_data.model_dump(),
                trace=trace
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            trace.end_ts = datetime.now()
            trace.latency_ms = latency_ms
            trace.errors.append(str(e))

            return SkillResponse(
                workspace=getattr(params, "workspace", "default"),
                ok=False,
                data={},
                trace=trace
            )

    @abstractmethod
    async def _execute_impl(self, params: InputT) -> OutputT:
        """
        Concrete implementation of the skill logic.
        Must return a Pydantic model representing the 'data' field.
        """
        pass
