import asyncio 
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from worker import ingest_document_task

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    job_id: str
    filename: str
    message: str


@router.post("/", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload a PDF or DOCX file and ingest it into the vector store.

    Uses plain def (not async def) so FastAPI automatically runs this in a
    thread pool executor — concurrent uploads are handled correctly without
    blocking the event loop. When Stage 4 (Celery) arrives, this handler will
    become a thin wrapper that enqueues a background task and returns a job ID.
    """

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Only .pdf and .docx are supported.",
        )

    # Save uploaded bytes to a temp file on disk — Docling needs a file path, not a stream
     # File I/O is blocking — run in a thread pool to avoid blocking the event loop
    def _save_to_temp() -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            return tmp.name

    tmp_path = await asyncio.to_thread(_save_to_temp)      # ← await + to_thread

    # Enqueue — returns immediately with a task ID
    # .delay() is a quick Redis put, safe to call directly from async
    task = ingest_document_task.delay(tmp_path, file.filename)

    
    return IngestResponse(
        job_id=task.id,
        filename=file.filename,
        message=f"Ingestion queued for {file.filename}. Poll /jobs/{task.id} for status.",
    )
