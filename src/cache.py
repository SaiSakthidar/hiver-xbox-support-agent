"""
Disk cache for LLM responses. Keyed on (backend, model, system, prompt).
Avoids repeat API calls during evaluation runs, keeping costs low and
making the 15-minute reproduction requirement reliable.

Usage: imported automatically by llm.py when CACHE_LLM=1 is set in .env
"""

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "outputs" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key(backend: str, model: str, system: str, prompt: str) -> str:
    raw = json.dumps([backend, model, system, prompt], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def get(backend: str, model: str, system: str, prompt: str) -> str | None:
    path = CACHE_DIR / (_key(backend, model, system, prompt) + ".txt")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def put(backend: str, model: str, system: str, prompt: str, response: str) -> None:
    path = CACHE_DIR / (_key(backend, model, system, prompt) + ".txt")
    path.write_text(response, encoding="utf-8")


def stats() -> dict:
    files = list(CACHE_DIR.glob("*.txt"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"entries": len(files), "size_kb": round(total_bytes / 1024, 1)}
