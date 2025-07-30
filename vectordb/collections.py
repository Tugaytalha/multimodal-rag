"""Wrapper to open/initialize Chroma collections for text, image caption, and page images."""
from __future__ import annotations

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from ingest.schema import CollectionNames

# Shared local embedding model (HuggingFace) -------------------------------------------------
class JinaV4LocalEmbeddings(Embeddings):
    """Local jina-embeddings-v4 wrapper that enforces `task='retrieval'`."""

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v4", **kwargs):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, **kwargs)

    def embed_documents(self, texts):
        return self.model.encode(texts, task="retrieval", convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text, task="retrieval", convert_to_numpy=True, show_progress_bar=False).tolist()


embedder = JinaV4LocalEmbeddings()

BASE_DIR = Path("chroma")


def get_collections(persist_root: Path | str = BASE_DIR):
    """Return (text_db, image_db, page_db) Chroma vector stores."""
    root = Path(persist_root)
    text_db = Chroma(persist_directory=str(root / CollectionNames.TEXT), embedding_function=embedder)
    img_db = Chroma(persist_directory=str(root / CollectionNames.IMAGE), embedding_function=embedder)
    page_db = Chroma(persist_directory=str(root / CollectionNames.PAGES), embedding_function=embedder)
    return text_db, img_db, page_db
