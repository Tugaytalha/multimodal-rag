"""Script to populate / update chroma collections from files in ./data"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings

import sys, pathlib
# Ensure project root on sys.path so `ingest` package is importable when running directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.append(PROJECT_ROOT.as_posix())

from ingest.loaders import load_pdf, load_docx
from ingest.schema import CollectionNames, Chunk
from ingest.vision_utils import caption_image
from tqdm import tqdm

from vectordb.collections import get_collections


def iter_files(data_dir: Path) -> List[Path]:
    for p in data_dir.iterdir():
        if p.suffix.lower() in {".pdf", ".docx"}:
            yield p


def main(data_dir: str = "data", chroma_root: str = "chroma"):
    data_path = Path(data_dir)
    images_dir = data_path / "extracted_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # vector stores
    text_db, img_db, page_db = get_collections(chroma_root)

    embedder = JinaEmbeddings(model_name="jina-embeddings-v4")

    for f in tqdm(list(iter_files(data_path)), desc="Processing files"):
        if f.suffix.lower() == ".pdf":
            chunks = load_pdf(f, images_dir)
        elif f.suffix.lower() == ".docx":
            chunks = load_docx(f, images_dir)
        else:
            continue

        # Split into collections
        text_chunks = [c for c in chunks if c.type == "text" or c.type == "table"]
        image_chunks = [c for c in chunks if c.type == "image"]
        page_chunks = [c for c in chunks if c.type == "page"]

        if text_chunks:
            text_db.add_documents(convert(text_chunks))
        if image_chunks:
            img_db.add_documents(convert(image_chunks))
        if page_chunks:
            page_db.add_documents(convert(page_chunks))


# ---------------------------------------------------------------------------

def convert(chunks: List[Chunk]):
    """Convert to langchain Documents."""
    docs = []
    for ch in chunks:
        docs.append(Document(page_content=ch.page_content, metadata=ch.metadata))
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="chroma")
    args = parser.parse_args()

    main(args.data, args.out)
