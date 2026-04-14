# AFTER
import asyncio 
from fastapi import APIRouter
from pydantic import BaseModel
from worker import celery_app
from typing import Any, Optional

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # PENDING | STARTED | SUCCESS | FAILURE
    result: Optional[Any] = None   # filled when status = SUCCESS
    error: Optional[str] = None    # filled when status = FAILURE
@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):                      # ← async def
    """Poll this endpoint after POST /ingest to check ingestion progress."""

    # .status and .result hit Redis — run in thread pool
    def _fetch_task_info():
        task = celery_app.AsyncResult(job_id)
        return task.status, task.result

    status, result = await asyncio.to_thread(_fetch_task_info)   # ← await + to_thread

    response = JobStatusResponse(job_id=job_id, status=status)

    if status == "SUCCESS":
        response.result = result
    elif status == "FAILURE":
        response.error = str(result)

    return response