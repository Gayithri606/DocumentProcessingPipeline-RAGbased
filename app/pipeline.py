import sys
import logging
from datetime import datetime

import pandas as pd
from timescale_vector.client import uuid_from_time
from langfuse import observe

from config.settings import get_settings
from database.vector_store import VectorStore
from services.document_processor import DocumentProcessor
from services.chunker import Chunker

get_settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class IngestionPipeline:
    """Orchestrates: DocumentProcessor → Chunker → VectorStore."""

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.chunker = Chunker()
        self.vector_store = VectorStore()
        self.vector_store.create_tables()


    @observe()
    def ingest(self, file_path: str, original_filename: str = None) -> int:
        """Ingest a PDF or DOCX file into the vector store.

        Args:
            file_path: Path to the file to ingest.
            original_filename: The original filename to store in metadata.
                When called from the API, this is the uploaded file's name.
                When called from the CLI, this is left as None and the name
                is inferred from file_path — so existing behaviour is unchanged.

        Returns:
            Number of chunks ingested.
        """
        logging.info(f"Starting ingestion: {file_path}")

        # Step 1: Extract text + doc object from file
        processed_doc = self.document_processor.process(file_path, original_filename=original_filename)
        logging.info(f"Extracted text from {processed_doc.metadata['filename']}")

        # Step 2: Split into structure-aware chunks
        chunks = self.chunker.split(processed_doc)
        logging.info(f"Split into {len(chunks)} chunks")

        # Step 3: Embed each chunk and build records
        texts = [chunk.content for chunk in chunks]
        embeddings = self.vector_store.get_embeddings_batch(texts)  # 1 API call for all chunks
        records = [
            {
                "id": str(uuid_from_time(datetime.now())),
                "metadata": chunk.metadata,
                "contents": chunk.raw_content,
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Step 4: Upsert into TimescaleDB
        df = pd.DataFrame(records)
        self.vector_store.upsert(df)
        logging.info(f"Upserted {len(records)} chunks into vector store")

        return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    pipeline = IngestionPipeline()
    count = pipeline.ingest(file_path)
    print(f"\n✅ Done — ingested {count} chunks from {file_path}")