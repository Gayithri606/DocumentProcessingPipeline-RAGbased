import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from celery import Celery
from celery.signals import worker_process_init
from config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_worker",
    broker=settings.redis.url,
    backend=settings.redis.url,
)

celery_app.conf.update(
    task_track_started=True,  # makes STARTED status visible when worker picks up the task
)


@worker_process_init.connect
def init_worker_process(**kwargs):
    """Pre-warm Docling's ML models in each forked worker process.

    Docling loads PyTorch/EasyOCR/OpenCV native libraries. Initialising
    DocumentConverter() *after* the fork (here) rather than *before* avoids
    the SIGABRT / fork-safety crash that occurs when those C extensions are
    inherited across a fork boundary (especially on macOS).
    """
    from docling.document_converter import DocumentConverter  # noqa: F401
    DocumentConverter()  # loads models once per worker process


@celery_app.task(bind=True)
def ingest_document_task(self, file_path: str, original_filename: str) -> dict:
    """Background task that runs the full ingestion pipeline.

    Imported inside the function to avoid circular imports at module load time.
    Cleans up the temp file after ingestion whether it succeeds or fails.
    """
    try:
        from pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        chunk_count = pipeline.ingest(file_path, original_filename=original_filename)
        return {"filename": original_filename, "chunks_ingested": chunk_count}
    finally:
        Path(file_path).unlink(missing_ok=True)  # always clean up temp file