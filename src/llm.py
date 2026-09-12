"""
Unified LLM client supporting three backends:
  - anthropic  : Claude via Anthropic API  (ANTHROPIC_API_KEY)
  - gemini     : Gemini via Google AI API  (GEMINI_API_KEY)
  - ollama     : any local model via Ollama (OLLAMA_MODEL, default: llama3.2)

Set LLM_BACKEND in your .env to pick one. Defaults to "anthropic".

Usage:
    from src.llm import chat
    response = chat("Classify this tweet: ...")
"""

import os
from dotenv import load_dotenv

load_dotenv()

BACKEND = os.getenv("LLM_BACKEND", "anthropic").lower()

# ---------- default model names per backend ----------
_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.0-flash",
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
}

MODEL = os.getenv("LLM_MODEL", _DEFAULTS.get(BACKEND, ""))


_CACHE_ENABLED = os.getenv("CACHE_LLM", "0") == "1"


def chat(
    prompt: str,
    system: str = "",
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Send a single-turn prompt and return the text response."""
    if _CACHE_ENABLED:
        from src.cache import get, put
        cached = get(BACKEND, MODEL, system, prompt)
        if cached is not None:
            return cached

    if BACKEND == "anthropic":
        result = _anthropic(prompt, system, max_tokens, temperature)
    elif BACKEND == "gemini":
        result = _gemini(prompt, system, max_tokens, temperature)
    elif BACKEND == "ollama":
        result = _ollama(prompt, system, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {BACKEND!r}. Choose anthropic / gemini / ollama")

    if _CACHE_ENABLED:
        from src.cache import put
        put(BACKEND, MODEL, system, prompt, result)

    return result


# ── Anthropic ────────────────────────────────────────────────────────────────

def _anthropic(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    import anthropic  # type: ignore

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=key)
    kwargs: dict = dict(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    msg = client.messages.create(**kwargs)
    return msg.content[0].text


# ── Gemini ───────────────────────────────────────────────────────────────────

def _gemini(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    import re
    import time
    from google import genai  # type: ignore
    from google.genai.errors import ServerError  # type: ignore

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=key)

    kwargs: dict = dict(
        model=MODEL,
        input=prompt,
        generation_config={"temperature": temperature},
    )
    if system:
        kwargs["system_instruction"] = system

    for attempt in range(6):
        try:
            interaction = client.interactions.create(**kwargs)
            return interaction.output_text or ""
        except Exception as e:
            err_str = str(e)
            # 503 server overload — short backoff
            if "503" in err_str or "UNAVAILABLE" in err_str:
                if attempt == 5:
                    raise
                wait = 2 ** attempt
                print(f"\n  Gemini 503, retrying in {wait}s…")
                time.sleep(wait)
            # 429 rate limit — parse suggested retry time or use 65s
            elif "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                if attempt == 5:
                    raise
                match = re.search(r"retry in ([\d.]+)s", err_str)
                wait = float(match.group(1)) + 2 if match else 65.0
                print(f"\n  Gemini rate limit, waiting {wait:.0f}s…")
                time.sleep(wait)
            else:
                raise

    return ""


# ── Ollama (local) ────────────────────────────────────────────────────────────

def _ollama(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    import json
    import urllib.request

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result.get("response", "")


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Backend : {BACKEND}")
    print(f"Model   : {MODEL}")
    reply = chat("Say 'hello world' and nothing else.")
    print(f"Response: {reply}")
