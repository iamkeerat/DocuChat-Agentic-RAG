"""FastAPI application for DocuChat.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.agents import AgentWorkflow, build_agent_workflow
from app.config import Settings, get_settings
from app.errors import InvalidDocumentError, LLMError
from app.rag import RAGService
from app.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse

logger = logging.getLogger(__name__)

APP_VERSION = "0.2.0"


def get_rag_service(request: Request) -> RAGService:
    """Resolve the per-process RAG service from application state."""

    service: RAGService | None = getattr(request.app.state, "rag", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The RAG service is not ready.",
        )
    return service


def create_app(
    settings: Settings | None = None,
    agent_workflow: AgentWorkflow | None = None,
) -> FastAPI:
    """Application factory.

    Injecting ``settings`` and ``agent_workflow`` keeps the app easy to test
    without touching real environment variables or external LLM providers.
    """

    resolved_settings = settings or get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        workflow = (
            agent_workflow
            if agent_workflow is not None
            else build_agent_workflow(resolved_settings)
        )
        service = RAGService(resolved_settings, workflow=workflow)
        try:
            if resolved_settings.auto_ingest:
                ingested = await run_in_threadpool(
                    service.ingest_directory, resolved_settings.documents_dir
                )
                logger.info(
                    "Startup ingestion indexed %d document(s) from %s",
                    len(ingested),
                    resolved_settings.documents_dir,
                )

            app.state.rag = service
            logger.info(
                "DocuChat started (environment=%s, collection=%s, llm=%s)",
                resolved_settings.environment,
                resolved_settings.collection_name,
                workflow.model_name if workflow is not None else "disabled",
            )
            yield
        finally:
            service.close()
            logger.info("DocuChat stopped")

    app = FastAPI(
        title="DocuChat",
        summary="Enterprise multi-agent RAG API for asking questions over PDFs.",
        description=(
            "Upload PDFs with `POST /ingest` or place them in the configured "
            "documents directory, then ask grounded questions with `POST /ask`. "
            "A Researcher Agent drafts an answer and a Critic Agent verifies it "
            "against the retrieved sources before it is returned."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request) -> HealthResponse:
        """Liveness probe used by Docker and orchestrators."""

        rag = get_rag_service(request)
        return HealthResponse(
            status="ok",
            version=APP_VERSION,
            environment=resolved_settings.environment,
            indexed_chunks=rag.count_chunks(),
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=["rag"],
        summary="Ask a question and verify the answer with the Critic Agent",
        responses={
            502: {"description": "The configured LLM or agent workflow failed."},
            503: {"description": "The RAG service has not finished starting."},
        },
    )
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        rag = get_rag_service(request)
        try:
            return await run_in_threadpool(rag.ask, payload.question)
        except LLMError as exc:
            logger.warning("Agent workflow failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI workflow is currently unavailable.",
            ) from exc

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["ingestion"],
        summary="Index a text-based PDF",
        responses={
            413: {"description": "The PDF exceeds the configured upload limit."},
            415: {"description": "The uploaded file is not a PDF."},
            422: {"description": "The PDF is encrypted or has no extractable text."},
        },
    )
    async def ingest(
        request: Request,
        file: UploadFile = File(  # noqa: B008 - FastAPI dependency default
            ..., description="A text-based PDF file."
        ),
    ) -> IngestResponse:
        rag = get_rag_service(request)
        # Never trust a client-supplied path; keep only the basename for metadata.
        filename = Path(file.filename or "upload.pdf").name.strip() or "upload.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are supported.",
            )

        max_bytes = resolved_settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        await file.close()

        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is empty.",
            )
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"PDF exceeds the {resolved_settings.max_upload_mb} MB upload limit.",
            )

        try:
            result = await run_in_threadpool(rag.ingest_pdf, data, filename)
        except InvalidDocumentError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        return IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            pages=result.pages,
            chunks_added=result.chunks_added,
        )

    return app


app = create_app()
