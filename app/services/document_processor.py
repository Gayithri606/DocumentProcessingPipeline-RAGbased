from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from langfuse import observe


@dataclass
class ProcessedDocument:
    text: str       # markdown export — useful for logging and debugging
    doc: Any        # raw Docling document object — needed by HybridChunker
    metadata: dict


class DocumentProcessor:
    """Converts PDF and DOCX files into clean text using Docling."""

    def __init__(self):
        self.converter = DocumentConverter()

    @observe()
    def process(self, file_path: str, original_filename: str = None) -> ProcessedDocument:
        """Extract text from a PDF or DOCX file.

        Args:
            file_path: Absolute or relative path to the file.
            original_filename: The original name of the file as provided by the
                caller (e.g. the uploaded filename from the API). When provided,
                this is stored in metadata instead of the temp file's name.
                Falls back to path.name if not provided — so existing callers
                like the CLI (pipeline.py) are unaffected.

        Returns:
            ProcessedDocument with extracted text, raw doc object, and file metadata.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in {".pdf", ".docx"}:
            raise ValueError(f"Unsupported file type: {path.suffix}. Only .pdf and .docx are supported.")

        result = self.converter.convert(file_path)

        metadata = {
            "filename": original_filename or path.name,  # use original name if provided, else infer from path
            "file_type": path.suffix.lower(),
            "file_path": str(path.resolve()),
        }

        return ProcessedDocument(
            text=result.document.export_to_markdown(),  # for logging
            doc=result.document,                         # for HybridChunker
            metadata=metadata,
        )