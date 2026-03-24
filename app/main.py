import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.vllm_engine import generate, init_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("SKIP_VLLM_INIT"):
        init_engine()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)


class InferResponse(BaseModel):
    model: str
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/infer", response_model=InferResponse)
async def infer(body: InferRequest):
    if os.getenv("SKIP_VLLM_INIT"):
        raise HTTPException(503, "vLLM not loaded (CI/test mode)")
    return InferResponse(model=settings.model_name, text=await generate(body.prompt))
