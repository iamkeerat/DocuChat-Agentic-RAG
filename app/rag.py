"""Retrieval, PDF ingestion, and the DocuChat agent hand-off.

The vector store is Chroma in embedded/persistent mode so a single container can
run without extra infrastructure. Everything I/O-bound and LLM-bound is
synchronous; the API layer executes it in a worker thread so the event loop
stays responsive.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.agents import AgentWorkflow
from app.config import Settings
from app.errors import InvalidDocumentError
from app.schemas import AgentStep, AskResponse, SourceChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")
_BATCH_SIZE = 500
_EMPTY_ANSWER = "I couldn't find any relevant information in the indexed documents."

Metadata = dict[str, str | int | float | bool]


class HashingEmbeddings(Embeddings):
    """Dependency-free hashed bag-of-words embeddings.

    This is a deliberate zero-configuration default: it needs no API key or model
    download, is deterministic, and keeps the demo runnable offline. Retrieval
    quality is lexical rather than semantic.

    For production, replace this with a real embedding model that implements the
    same LangChain ``Embeddings`` interface (for example OpenAI
    ``text-embedding-3-small`` or a local BGE model). Keep the dimension stable
    per collection; changing it requires re-indexing into a new collection.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall(text.casefold())
        vector = [0.0] * self.dimensions
        if not tokens:
            return vector

        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    filename: str
    pages: int
    chunks_added: int


class RAGService:
    """Owns the PDF ingestion pipeline, Chroma collection, and agent workflow."""

    def __init__(
        self,
        settings: Settings,
        workflow: AgentWorkflow | None = None,
    ) -> None:
        self._settings = settings
        self._workflow = workflow
        self._embeddings = HashingEmbeddings(settings.embedding_dimensions)

        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Chroma's embedded client is not designed for concurrent writers.
        self._write_lock = threading.Lock()

    def ingest_pdf(self, data: bytes, filename: str) -> IngestResult:
        """Extract, chunk, embed, and store a PDF.

        Chunks are keyed deterministically by document hash + page + chunk index,
        and older chunks for the same document are replaced. Re-loading the same
        file (including on every startup) therefore does not duplicate vectors.

        Raises:
            InvalidDocumentError: if the file is not a readable, text-based PDF.
        """

        if not data.startswith(b"%PDF-"):
            raise InvalidDocumentError("The uploaded file is not a valid PDF.")

        try:
            reader = PdfReader(BytesIO(data))
            page_count = len(reader.pages)
            is_encrypted = reader.is_encrypted
        except Exception as exc:
            logger.warning("Failed to parse PDF %s: %s", filename, exc)
            raise InvalidDocumentError("The PDF could not be parsed.") from exc

        if is_encrypted:
            raise InvalidDocumentError("Encrypted PDFs are not supported.")
        if page_count == 0:
            raise InvalidDocumentError("The PDF contains no pages.")

        document_id = hashlib.sha256(data).hexdigest()[:16]
        texts: list[str] = []
        metadatas: list[Metadata] = []
        ids: list[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                logger.exception(
                    "Failed to extract text from %s (page %d)", filename, page_number
                )
                continue
            if not page_text.strip():
                continue

            for chunk_index, chunk in enumerate(self._splitter.split_text(page_text)):
                chunk = chunk.strip()
                if chunk:
                    texts.append(chunk)
                    metadatas.append(
                        {
                            "source": filename,
                            "page": page_number,
                            "document_id": document_id,
                        }
                    )
                    ids.append(f"{document_id}:{page_number}:{chunk_index}")

        if not texts:
            raise InvalidDocumentError(
                "No extractable text was found. The PDF may be scanned or "
                "image-only; OCR is not configured."
            )

        # Replace any previous version of this exact document. This makes both
        # startup loading and repeated /ingest calls idempotent.
        with self._write_lock:
            self._collection.delete(where={"document_id": document_id})
            self._add_texts(texts, metadatas, ids)
        return IngestResult(
            document_id=document_id,
            filename=filename,
            pages=page_count,
            chunks_added=len(texts),
        )

    def ingest_directory(self, directory: Path) -> list[IngestResult]:
        """Index every PDF in ``directory``, logging and skipping bad files."""

        if not directory.is_dir():
            logger.info(
                "Documents directory %s does not exist; skipping startup ingestion.",
                directory,
            )
            return []

        results: list[IngestResult] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            try:
                result = self.ingest_pdf(path.read_bytes(), path.name)
            except (OSError, InvalidDocumentError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
            results.append(result)
            logger.info("Indexed %s (%d chunks).", path.name, result.chunks_added)
        return results

    def ask(self, question: str) -> AskResponse:
        """Retrieve relevant chunks and run the Researcher/Critic workflow."""

        sources = self._retrieve(question)
        if not sources:
            return AskResponse(
                answer=_EMPTY_ANSWER,
                verified=False,
                confidence=None,
                attempts=0,
                critic_feedback="No relevant documents were retrieved.",
                sources=[],
                model="none",
                workflow=[],
            )

        if self._workflow is None:
            return AskResponse(
                answer=self._extractive_answer(sources),
                verified=False,
                confidence=None,
                attempts=0,
                critic_feedback=(
                    "No LLM is configured; this is a retrieval-only fallback, "
                    "not a verified answer."
                ),
                sources=sources,
                model="extractive",
                workflow=[
                    AgentStep(
                        agent="researcher",
                        action="extractive_fallback",
                        status="completed",
                        detail="Returned the top retrieved passages without an LLM.",
                        duration_ms=0,
                    )
                ],
            )

        result = self._workflow.run(question, sources)
        return AskResponse(
            answer=result.answer,
            verified=result.verified,
            confidence=result.confidence,
            attempts=result.attempts,
            critic_feedback=result.critic_feedback,
            sources=sources,
            model=self._workflow.model_name,
            workflow=result.workflow,
        )

    def count_chunks(self) -> int:
        """Return the number of vectors currently stored."""

        return int(self._collection.count())

    def close(self) -> None:
        """Release resources.

        Chroma's persistent client does not require an explicit close, and the
        LangChain chat model is process-scoped. The hook exists for future
        providers and for symmetrical lifespan cleanup.
        """

    def _add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[Metadata],
        ids: Sequence[str],
    ) -> None:
        for start in range(0, len(texts), _BATCH_SIZE):
            batch_texts = list(texts[start : start + _BATCH_SIZE])
            batch_metadatas = list(metadatas[start : start + _BATCH_SIZE])
            batch_ids = list(ids[start : start + _BATCH_SIZE])
            self._collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=self._embeddings.embed_documents(batch_texts),
            )

    def _retrieve(self, question: str) -> list[SourceChunk]:
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(question)],
            n_results=self._settings.retrieval_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        sources: list[SourceChunk] = []
        for content, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            page = metadata.get("page")
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            sources.append(
                SourceChunk(
                    content=content,
                    source=str(metadata.get("source", "unknown")),
                    page=page if isinstance(page, int) and page >= 1 else None,
                    score=round(similarity, 4),
                )
            )
        return sources

    @staticmethod
    def _extractive_answer(sources: Sequence[SourceChunk]) -> str:
        passages = []
        for index, source in enumerate(sources[:2], start=1):
            text = " ".join(source.content.split())
            if len(text) > 700:
                text = f"{text[:697].rstrip()}..."
            passages.append(f"[{index}] {text}")

        return (
            "No LLM is configured, so DocuChat is returning the most relevant "
            "indexed passages instead.\n\n" + "\n\n".join(passages)
        )
