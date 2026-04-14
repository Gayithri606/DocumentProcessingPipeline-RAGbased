from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.vector_store import VectorStore
from services.synthesizer import Synthesizer

router = APIRouter(prefix="/query", tags=["query"])

# Created once at startup and shared across all requests.
# VectorStore holds a DB connection — no need to recreate it per request.
vector_store = VectorStore()


class QueryRequest(BaseModel):
    question: str
    limit: int = 5


class QueryResponse(BaseModel):
    answer: str
    thought_process: List[str]
    enough_context: bool


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question and get an answer synthesized from the ingested documents.

    Uses async def so FastAPI runs this on the event loop — I/O operations
    (embedding call, DB search, LLM synthesis) are all awaited without
    blocking the event loop or occupying a thread-pool slot.
    """

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    results = await vector_store.search(request.question, limit=request.limit)
    response = await Synthesizer.generate_response(
        question=request.question,
        context=results,
    )

    return QueryResponse(
        answer=response.answer,
        thought_process=response.thought_process,
        enough_context=response.enough_context,
    )
