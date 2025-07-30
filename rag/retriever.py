"""Hybrid retriever merging text & image/page results."""
from __future__ import annotations

from typing import List

from langchain.schema import Document

from vectordb.collections import get_collections


def hybrid_retrieve(query: str, k_text: int = 8, k_page: int = 2):
    text_db, img_db, page_db = get_collections()

    text_hits = text_db.similarity_search_with_score(query, k=k_text)
    page_hits = page_db.similarity_search_with_score(query, k=k_page)

    docs: List[Document] = []
    for d, score in text_hits + page_hits:
        d.metadata["score"] = score
        docs.append(d)
    docs.sort(key=lambda x: x.metadata["score"], reverse=False)
    return docs
