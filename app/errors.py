"""Shared application exceptions."""

from __future__ import annotations


class RAGError(Exception):
    """Base class for expected DocuChat failures."""


class InvalidDocumentError(RAGError):
    """The uploaded document cannot be parsed or contains no usable text."""


class LLMError(RAGError):
    """The configured language model or agent workflow failed."""
