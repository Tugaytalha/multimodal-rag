"""Pydantic data structures shared across the multimodal-rag project."""
from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable chunk of any modality (text, image, table, page)."""

    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Convenience helpers ---------------------------------------------------
    @property
    def id(self) -> str:
        return self.metadata.get("id", "")

    @property
    def type(self) -> str:
        return self.metadata.get("type", "text")


class CollectionNames:
    """Logical names for the three Chroma collections we maintain."""

    TEXT = "chroma_text"
    IMAGE = "chroma_image"
    PAGES = "chroma_pages"
