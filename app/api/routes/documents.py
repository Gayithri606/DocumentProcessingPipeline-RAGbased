from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.vector_store import VectorStore

router = APIRouter(prefix="/documents", tags=["documents"])

# Created once at startup and shared across all requests.
vector_store = VectorStore()


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


@router.get("/", response_model=DocumentsResponse)
async def list_documents():
    """List all documents that have been ingested into the vector store,
    along with their file type and how many chunks they were split into.
    """

    try:
        docs = await vector_store.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DocumentsResponse(
        documents=[
            DocumentInfo(
                filename=d["filename"],
                file_type=d["file_type"],
                chunk_count=d["chunk_count"],
            )
            for d in docs
        ],
        total=len(docs),
    )
