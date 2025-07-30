"""Helpers for image handling & captioning via Gemma-3 4B (Ollama)."""
from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import List

from PIL import Image

OLLAMA_HOST = "http://localhost:11434"  # default
VLM_MODEL = "gemma3:4b-vision"


def caption_image(img: Image.Image, prompt: str = "Describe the image succinctly.") -> str:
    """Call Ollama Gemma3 4B vision model via subprocess HTTP streaming."""
    import requests, base64, json

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()

    payload = {
        "model": VLM_MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["response"].strip()


def save_image_bytes(img_bytes: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(img_bytes)).save(out_path)
