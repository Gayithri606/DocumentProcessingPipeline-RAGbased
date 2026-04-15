from dataclasses import dataclass
from typing import List
import logging

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from langfuse import observe

from config.settings import get_settings
from services.document_processor import ProcessedDocument


@dataclass
class TextChunk:
    content: str        # contextualized text → used for embedding (includes parent headings)
    raw_content: str    # raw chunk text → stored in DB and shown to users
    metadata: dict


class Chunker:
    """Splits Docling documents into structure-aware chunks using HybridChunker."""

    def __init__(self):
        settings = get_settings().chunking
        self.max_tokens = settings.max_tokens
        self.chunk_max_tokens = settings.max_tokens - settings.heading_token_reserve
        tiktoken_encoder = tiktoken.encoding_for_model(settings.embedding_model)
        self.tokenizer = OpenAITokenizer(
            tokenizer=tiktoken_encoder,
            max_tokens=self.chunk_max_tokens,  # leaves headroom for contextualize() headings
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer)

    @observe()
    def split(self, processed_doc: ProcessedDocument) -> List[TextChunk]:
        """Split a processed document into semantically-aware chunks.

        Uses HybridChunker for document-structure-aware chunking.
        Each chunk has two text versions:
          - raw_content: the actual chunk text (stored in DB, shown to users)
          - content: contextualized text with parent headings added (used for embedding)

        Args:
            processed_doc: Output from DocumentProcessor.process()

        Returns:
            List of TextChunk objects.
        """
        raw_chunks = list(self.chunker.chunk(dl_doc=processed_doc.doc))
        chunks = []

        for i, chunk in enumerate(raw_chunks):
            raw_text = chunk.text
            contextualized_text = self.chunker.contextualize(chunk=chunk)

            # Safety net: heading_token_reserve should prevent this from ever
            # firing, but if headings are unusually long and still push past
            # the limit, fall back to raw text rather than truncating — the
            # full chunk content is always preserved this way.
            contextualized_token_count_raw = len(self.tokenizer.tokenizer.encode(contextualized_text))
            if contextualized_token_count_raw > self.max_tokens:
                logging.warning(
                    f"Chunk {i}: contextualized text ({contextualized_token_count_raw} tokens) "
                    f"exceeds limit ({self.max_tokens}) despite reserve — "
                    f"falling back to raw text for embedding"
                )
                contextualized_text = raw_text

            raw_token_count = len(self.tokenizer.tokenizer.encode(raw_text))
            contextualized_token_count = len(self.tokenizer.tokenizer.encode(contextualized_text))

            logging.info(f"Chunk {i}: {raw_token_count} raw tokens, {contextualized_token_count} contextualized tokens")

            chunk_metadata = {
                **processed_doc.metadata,
                "chunk_index": i,
                "raw_token_count": len(self.tokenizer.tokenizer.encode(raw_text)),
                "contextualized_token_count": len(self.tokenizer.tokenizer.encode(contextualized_text)),
            }

            chunks.append(TextChunk(
                content=contextualized_text,  # richer text → better embeddings
                raw_content=raw_text,          # clean text → stored in DB
                metadata=chunk_metadata,
            ))

        return chunks