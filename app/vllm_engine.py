from __future__ import annotations

import uuid

from app.config import settings

_engine = None


def init_engine() -> None:
    global _engine  # noqa: PLW0603
    if _engine is not None:
        return
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    _engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=settings.model_name,
            trust_remote_code=settings.trust_remote_code,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            enable_lora=bool(settings.lora_path),
            max_lora_rank=64,
        )
    )


def _get_engine():
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


async def generate(prompt: str) -> str:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    lora = LoRARequest("adapter", 1, settings.lora_path) if settings.lora_path else None
    sp = SamplingParams(temperature=settings.temperature, max_tokens=settings.max_tokens)
    text = ""
    async for ro in _get_engine().generate(prompt, sp, str(uuid.uuid4()), lora_request=lora):
        for o in ro.outputs:
            text = o.text  # cumulative; last value is the complete output
    return text
