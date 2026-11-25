"""API, ingestion, and multi-agent workflow tests.

The agent tests use LangChain's scripted fake chat model, so they exercise the
real prompt/parser/orchestration code without requiring a Groq key or Ollama.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agents import AgentResult, MultiAgentRAG
from app.config import Settings
from app.errors import LLMError
from app.rag import HashingEmbeddings, RAGService
from app.schemas import SourceChunk
from main import create_app

_VERSION = "0.2.0"


def _settings(
    tmp_path: Path,
    *,
    auto_ingest: bool = False,
    documents_dir: Path | None = None,
) -> Settings:
    return Settings(
        environment="test",
        chroma_dir=tmp_path / "chroma",
        documents_dir=documents_dir or tmp_path / "documents",
        auto_ingest=auto_ingest,
        collection_name="docuchat_test",
        embedding_dimensions=128,
        chunk_size=200,
        chunk_overlap=20,
        retrieval_k=3,
        max_upload_mb=1,
        critic_max_revisions=1,
    )


def _build_pdf(text: str) -> bytes:
    """Create a tiny single-page PDF without adding a PDF-writing dependency."""

    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(output)


def _source() -> list[SourceChunk]:
    return [
        SourceChunk(
            content="Enterprise support has a four-hour response SLA.",
            source="sla.pdf",
            page=1,
            score=0.91,
        )
    ]


def _approved_review() -> str:
    return (
        '{"is_supported": true, "confidence": 0.95, '
        '"issues": [], "unsupported_claims": []}'
    )


def _flagged_review(issue: str) -> str:
    return (
        '{"is_supported": false, "confidence": 0.2, '
        f'"issues": ["{issue}"], "unsupported_claims": ["{issue}"]}}'
    )


def _workflow(responses: list[str], max_revisions: int = 1) -> MultiAgentRAG:
    return MultiAgentRAG(
        llm=FakeListChatModel(responses=responses),
        model_name="fake:scripted",
        max_revisions=max_revisions,
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": _VERSION,
        "environment": "test",
        "indexed_chunks": 0,
    }


def test_settings_resolve_groq_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.llm_provider == "groq"
    assert settings.llm_api_key == "gsk_test"
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.llm_model == "llama-3.1-8b-instant"


def test_settings_supply_ollama_dummy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.llm_provider == "ollama"
    assert settings.llm_api_key == "ollama"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "llama3.2"


def test_ask_on_empty_index(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is the refund policy?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"] == []
    assert payload["model"] == "none"
    assert payload["verified"] is False
    assert payload["attempts"] == 0
    assert "couldn't find" in payload["answer"]
    assert "No relevant documents" in payload["critic_feedback"]
def test_ask_returns_200_with_answer_key(client: TestClient) -> None:
    """Core API contract: POST /ask returns 200 and an answer field."""

    response = client.post(
        "/ask",
        json={"question": "What does the document say about refunds?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert isinstance(payload["answer"], str)




@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2_001},
        {"question": "Valid question", "unexpected": "field"},
    ],
)
def test_ask_rejects_invalid_payloads(client: TestClient, payload: dict[str, str]) -> None:
    response = client.post("/ask", json=payload)

    assert response.status_code == 422


def test_ingest_then_ask_returns_sources(client: TestClient) -> None:
    pdf = _build_pdf(
        "DocuChat refund policy: customers may request a refund within 30 days of purchase."
    )

    ingest_response = client.post(
        "/ingest",
        files={"file": ("policy.pdf", pdf, "application/pdf")},
    )
    assert ingest_response.status_code == 201, ingest_response.text
    ingested = ingest_response.json()
    assert ingested["filename"] == "policy.pdf"
    assert ingested["pages"] == 1
    assert ingested["chunks_added"] >= 1
    assert len(ingested["document_id"]) == 16
    assert client.get("/health").json()["indexed_chunks"] == ingested["chunks_added"]

    ask_response = client.post(
        "/ask",
        json={"question": "How many days do customers have to request a refund?"},
    )
    assert ask_response.status_code == 200, ask_response.text
    answered = ask_response.json()
    assert answered["model"] == "extractive"
    assert answered["verified"] is False
    assert answered["sources"]
    assert answered["sources"][0]["source"] == "policy.pdf"
    assert answered["sources"][0]["page"] == 1
    assert 0.0 <= answered["sources"][0]["score"] <= 1.0
    assert "30 days" in answered["answer"]


def test_startup_ingests_local_pdf(tmp_path: Path) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "document.pdf").write_bytes(
        _build_pdf("The enterprise escalation contact is support@docuchat.example.")
    )

    app = create_app(
        _settings(tmp_path, auto_ingest=True, documents_dir=documents_dir)
    )
    with TestClient(app) as test_client:
        health = test_client.get("/health").json()
        assert health["indexed_chunks"] >= 1
        response = test_client.post(
            "/ask", json={"question": "What is the enterprise escalation contact?"}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"][0]["source"] == "document.pdf"


def test_reingesting_same_pdf_replaces_chunks(tmp_path: Path) -> None:
    service = RAGService(_settings(tmp_path))
    pdf = _build_pdf("The support SLA is four hours.")
    try:
        first = service.ingest_pdf(pdf, "document.pdf")
        chunk_count = service.count_chunks()
        second = service.ingest_pdf(pdf, "document.pdf")
        final_chunk_count = service.count_chunks()
    finally:
        service.close()

    assert first.document_id == second.document_id
    assert final_chunk_count == chunk_count
    assert first.chunks_added == second.chunks_added


def test_critic_approves_supported_answer() -> None:
    workflow = _workflow(
        ["Enterprise support has a four-hour response SLA [1].", _approved_review()]
    )

    result = workflow.run("What is the support SLA?", _source())

    assert result.verified is True
    assert result.confidence == 0.95
    assert result.attempts == 1
    assert result.critic_feedback is None
    assert [step.agent for step in result.workflow] == ["researcher", "critic"]
    assert result.workflow[-1].status == "approved"


def test_critic_flags_hallucination_and_forces_revision() -> None:
    workflow = _workflow(
        [
            "The support SLA is 24 hours [1].",
            _flagged_review("The source says four hours, not 24 hours."),
            "The support SLA is four hours [1].",
            _approved_review(),
        ]
    )

    result = workflow.run("What is the support SLA?", _source())

    assert result.verified is True
    assert result.attempts == 2
    assert [step.action for step in result.workflow] == [
        "draft",
        "review",
        "revision",
        "review",
    ]
    assert result.workflow[1].status == "flagged"
    assert result.workflow[-1].status == "approved"


def test_critic_withholds_answer_after_revision_limit() -> None:
    workflow = _workflow(
        [
            "The support SLA is 24 hours [1].",
            _flagged_review("The source says four hours, not 24 hours."),
            "The support SLA is 12 hours [1].",
            _flagged_review("The source says four hours, not 12 hours."),
        ]
    )

    result = workflow.run("What is the support SLA?", _source())

    assert result.verified is False
    assert result.attempts == 2
    assert "couldn't verify an answer" in result.answer
    assert "four hours" in result.critic_feedback
    assert result.workflow[-1].status == "fallback"


def test_unparseable_critic_response_fails_closed() -> None:
    workflow = _workflow(["The support SLA is four hours [1].", "not json"])

    result = workflow.run("What is the support SLA?", _source())

    assert result.verified is False
    assert result.attempts == 1
    assert "could not be parsed" in result.critic_feedback
    assert any(step.status == "flagged" for step in result.workflow)
    assert result.workflow[-1].status == "fallback"


def test_ask_uses_injected_agent_workflow(tmp_path: Path) -> None:
    workflow = _workflow(
        ["Enterprise support has a four-hour response SLA [1].", _approved_review()]
    )
    app = create_app(_settings(tmp_path), agent_workflow=workflow)
    pdf = _build_pdf("Enterprise support has a four-hour response SLA.")

    with TestClient(app) as test_client:
        assert (
            test_client.post(
                "/ingest", files={"file": ("sla.pdf", pdf, "application/pdf")}
            ).status_code
            == 201
        )
        response = test_client.post("/ask", json={"question": "What is the SLA?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["verified"] is True
    assert payload["model"] == "fake:scripted"
    assert payload["attempts"] == 1
    assert [step["agent"] for step in payload["workflow"]] == ["researcher", "critic"]


class _FailingWorkflow:
    model_name = "fake:failing"

    def run(self, question: str, sources: Sequence[SourceChunk]) -> AgentResult:
        raise LLMError("upstream unavailable")


def test_llm_failure_maps_to_502(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), agent_workflow=_FailingWorkflow())
    pdf = _build_pdf("The support SLA for enterprise customers is four hours.")

    with TestClient(app) as test_client:
        test_client.post("/ingest", files={"file": ("sla.pdf", pdf, "application/pdf")})
        response = test_client.post("/ask", json={"question": "What is the SLA?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "The AI workflow is currently unavailable."


def test_ingest_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 415


def test_ingest_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_ingest_rejects_oversized_file(client: TestClient) -> None:
    oversized = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 1)

    response = client.post(
        "/ingest",
        files={"file": ("large.pdf", oversized, "application/pdf")},
    )

    assert response.status_code == 413


def test_ingest_rejects_malformed_pdf(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("broken.pdf", b"%PDF-1.4\nnot really a pdf", "application/pdf")},
    )

    assert response.status_code == 422


def test_ingest_strips_path_from_filename(client: TestClient) -> None:
    pdf = _build_pdf("The escalation contact is support@docuchat.example.")

    response = client.post(
        "/ingest",
        files={"file": ("../../etc/contact.pdf", pdf, "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "contact.pdf"


def test_hashing_embeddings_are_deterministic() -> None:
    embeddings = HashingEmbeddings(dimensions=64)

    first = embeddings.embed_query("refund policy")
    second = embeddings.embed_query("refund policy")

    assert first == second
    assert len(first) == 64
    assert any(value != 0.0 for value in first)
