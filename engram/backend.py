"""
backend.py — the only file that talks to the outside world.

Wraps the LLM and embedding calls. Default backend is Ollama (local, free).
Swap these two functions to use any provider; nothing else in Engram needs
to change.

Setup:
    install Ollama from https://ollama.com
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    pip install ollama
"""

import json
import re

LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"


def llm(prompt: str) -> str:
    """Single-shot completion. Low temperature for steadier JSON."""
    import ollama
    r = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return r["message"]["content"]


def embed(text: str) -> list[float]:
    """Return an embedding vector for `text`."""
    import ollama
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def parse_json(text: str) -> dict:
    """
    Defensive JSON extraction.

    Small local models leak prose and code fences around their JSON, so we
    grab the first {...} block and tolerate failure. A malformed consolidation
    degrades to 'nothing learned this block' — the chain still advances and
    continuity holds. Graceful failure matters when the model is 3B.
    """
    empty = {"new_episodes": [], "reinforce": [], "supersede": []}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return empty
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return empty
    # ensure all keys exist so callers never KeyError
    for k in empty:
        out.setdefault(k, [])
    return out