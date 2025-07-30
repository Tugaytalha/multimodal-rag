"""Loaders for PDF, DOCX, images; produce Chunk objects."""
from __future__ import annotations

import io
import fitz  # pymupdf
from pathlib import Path
from typing import List, Dict

from pdf2image import convert_from_path
from docx import Document as DocxDocument  # python-docx
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .schema import Chunk
from .vision_utils import caption_image, save_image_bytes


splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def load_pdf(path: str | Path, images_dir: Path) -> List[Chunk]:
    pdf_path = Path(path)
    doc = fitz.open(pdf_path)
    seq = 0
    chunks: List[Chunk] = []
    for page_idx, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        saw_image = False
        for blk in blocks:
            if blk["type"] == 0:  # text
                raw = " ".join(span["text"] for line in blk["lines"] for span in line["spans"]).strip()
                if not raw:
                    continue
                for txt in splitter.split_text(raw):
                    chunks.append(Chunk(page_content=txt, metadata={
                        "type": "text",
                        "page": page_idx,
                        "sequence_id": seq,
                        "source_file": pdf_path.name,
                    }))
                    seq += 1
            elif blk["type"] == 1:  # image
                saw_image = True
                try:
                    img_bytes = _safe_block_image(page, doc, blk)
                    img_path = images_dir / f"{pdf_path.stem}_seq{seq:04d}.png"
                    save_image_bytes(img_bytes, img_path)
                    caption = caption_image(Image.open(io.BytesIO(img_bytes)))
                    chunks.append(Chunk(page_content=caption, metadata={
                        "type": "image",
                        "page": page_idx,
                        "sequence_id": seq,
                        "image_path": str(img_path),
                        "source_file": pdf_path.name,
                    }))
                    seq += 1
                except Exception:
                    pass
        # if page had no explicit images, rasterise whole page for page collection
        if not saw_image:
            full_bytes = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
            img_path = images_dir / f"{pdf_path.stem}_page{page_idx:03d}.png"
            save_image_bytes(full_bytes, img_path)
            chunks.append(Chunk(page_content="PAGE_IMAGE_PLACEHOLDER", metadata={
                "type": "page",
                "page": page_idx,
                "image_path": str(img_path),
                "source_file": pdf_path.name,
            }))
    doc.close()
    return chunks


def _safe_block_image(page, doc, blk, zoom: float = 2.0):
    xref = blk.get("xref", 0)
    if isinstance(xref, int) and xref > 0:
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            return pix.tobytes("png")
        except Exception:
            pass
    x0, y0, x1, y1 = blk["bbox"]
    rect = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png")


def load_docx(path: str | Path, images_dir: Path) -> List[Chunk]:
    doc_path = Path(path)
    document = DocxDocument(doc_path)
    full_text = []
    for para in document.paragraphs:
        full_text.append(para.text)
    chunks = [Chunk(page_content=t, metadata={
        "type": "text",
        "page": 1,
        "source_file": doc_path.name,
    }) for t in splitter.split_text("\n".join(full_text))]

    # render pages via docx→pdf→image skipped for brevity
    return chunks
