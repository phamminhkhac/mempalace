"""
embeddings.py — Pluggable embedding backends for MemPalace.

Supports four providers:
  - "default"    — ChromaDB built-in (all-MiniLM-L6-v2, ONNX-based)
  - "ollama"     — Local Ollama server (bge-m3 default, zero extra deps)
  - "llama-cpp"  — Local llama.cpp server with OpenAI-compatible API
  - "gemini"     — Google Gemini Embedding API (requires google-genai)

The factory returns None for "default", letting ChromaDB use its own model.
"""

import atexit
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from typing import Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from .config import MempalaceConfig

logger = logging.getLogger("mempalace")

# ---------------------------------------------------------------------------
# Ollama — calls /api/embed via stdlib, zero extra dependencies
# ---------------------------------------------------------------------------


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding via a running Ollama instance."""

    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def __call__(self, input: Documents) -> Embeddings:
        payload = json.dumps({"model": self._model, "input": input}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}: {exc.reason}\n"
                f"Ensure the model is pulled: ollama pull {self._model}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url} — is it running?\n"
                f"Start with: ollama serve\n"
                f"Then pull the model: ollama pull {self._model}"
            ) from exc
        return data["embeddings"]


# ---------------------------------------------------------------------------
# llama.cpp — OpenAI-compatible /v1/embeddings, zero extra dependencies
# ---------------------------------------------------------------------------


class LlamaCppEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding via a llama.cpp server (llama-server --embedding)."""

    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:8080"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    # bge-m3 context is 8192 tokens; ~4 chars/token → truncate at ~30k chars
    MAX_CHARS = 30_000

    def __call__(self, input: Documents) -> Embeddings:
        truncated = [doc[:self.MAX_CHARS] if len(doc) > self.MAX_CHARS else doc for doc in input]
        payload = json.dumps({"input": truncated, "model": self._model}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError:
            # Server returned 500 (e.g. input still too large) — retry with
            # aggressive truncation to avoid crashing llama-server.
            shorter = [doc[:8_000] for doc in truncated]
            payload2 = json.dumps({"input": shorter, "model": self._model}).encode()
            req2 = urllib.request.Request(
                f"{self._base_url}/v1/embeddings",
                data=payload2,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req2, timeout=120) as resp2:
                    data = json.loads(resp2.read())
            except urllib.error.URLError as exc:
                raise ConnectionError(
                    f"Cannot connect to llama.cpp server at {self._base_url}\n"
                    f"Start with: llama-server --model <gguf-file> --embedding --port 8080 --n-gpu-layers 0"
                ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot connect to llama.cpp server at {self._base_url}\n"
                f"Start with: llama-server --model <gguf-file> --embedding --port 8080 --n-gpu-layers 0"
            ) from exc
        return [item["embedding"] for item in data["data"]]


# ---------------------------------------------------------------------------
# Gemini — requires pip install google-genai
# ---------------------------------------------------------------------------


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding via Google Gemini API."""

    def __init__(self, model: str = "gemini-embedding-001", api_key: str = ""):
        try:
            from google import genai  # noqa: F401
        except ImportError:
            raise ImportError(
                "Gemini embeddings require the google-genai package.\n"
                "Install with: pip install mempalace[gemini]"
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        result = self._client.models.embed_content(model=self._model, contents=input)
        return [list(e.values) for e in result.embeddings]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedding_function(config: Optional[MempalaceConfig] = None) -> Optional[EmbeddingFunction]:
    """Return the configured EmbeddingFunction, or None for ChromaDB default."""
    config = config or MempalaceConfig()
    provider = config.embed_provider

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = config.embed_model or "bge-m3"
        logger.debug("Using Ollama embedding: model=%s url=%s", model, base_url)
        return OllamaEmbeddingFunction(model=model, base_url=base_url)

    if provider == "llama-cpp":
        base_url = os.environ.get("LLAMA_CPP_BASE_URL", "http://localhost:8080")
        model = config.embed_model or "bge-m3"
        logger.debug("Using llama.cpp embedding: model=%s url=%s", model, base_url)
        return LlamaCppEmbeddingFunction(model=model, base_url=base_url)

    if provider == "gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise ValueError(
                "Gemini embedding requires an API key.\n"
                "Set GEMINI_API_KEY env var or gemini_api_key in ~/.mempalace/config.json"
            )
        model = config.embed_model or "gemini-embedding-001"
        logger.debug("Using Gemini embedding: model=%s", model)
        return GeminiEmbeddingFunction(model=model, api_key=api_key)

    # "default" — let ChromaDB use its built-in ONNX model
    return None


# ---------------------------------------------------------------------------
# Collection helpers — centralized embedding injection
# ---------------------------------------------------------------------------

_UNSET: object = object()
_ef_cache: object = _UNSET
_ef_config_key: Optional[str] = None


def _get_ef(config: Optional[MempalaceConfig] = None) -> Optional[EmbeddingFunction]:
    """Cache the embedding function for the lifetime of the process."""
    global _ef_cache, _ef_config_key
    config = config or MempalaceConfig()
    key = f"{config.embed_provider}:{config.embed_model}"
    if _ef_cache is _UNSET or _ef_config_key != key:
        _ef_cache = get_embedding_function(config)
        _ef_config_key = key
    return _ef_cache  # type: ignore[return-value]


def get_collection(
    client: chromadb.ClientAPI,
    name: str,
    *,
    create: bool = False,
    config: Optional[MempalaceConfig] = None,
):
    """Get (or create) a ChromaDB collection with the configured embedding function."""
    ef = _get_ef(config)
    kwargs: dict = {"name": name}
    if ef is not None:
        kwargs["embedding_function"] = ef
    if create:
        return client.get_or_create_collection(**kwargs)
    return client.get_collection(**kwargs)


# ---------------------------------------------------------------------------
# llama-server process manager — auto start/stop with MCP lifecycle
# ---------------------------------------------------------------------------

_llama_proc: Optional[subprocess.Popen] = None
_atexit_registered: bool = False


def _find_model_path(config: Optional[MempalaceConfig] = None) -> Optional[str]:
    """Find the bge-m3 GGUF file from Ollama's cache."""
    manifest = os.path.expanduser(
        "~/.ollama/models/manifests/registry.ollama.ai/library/bge-m3/latest"
    )
    if not os.path.exists(manifest):
        return None
    try:
        with open(manifest) as f:
            data = json.load(f)
        for layer in data.get("layers", []):
            if "model" in layer.get("mediaType", ""):
                digest = layer["digest"].replace("sha256:", "sha256-")
                path = os.path.expanduser(f"~/.ollama/models/blobs/{digest}")
                if os.path.exists(path):
                    return path
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None


def _is_server_ready(base_url: str, timeout: float = 1.0) -> bool:
    """Check if a server is responding at base_url."""
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def start_llama_server(config: Optional[MempalaceConfig] = None, port: int = 0) -> Optional[str]:
    """Start a llama-server subprocess if llama-cpp provider is configured.

    Returns the base_url if started, or None if not needed / already running.
    The process is killed automatically on interpreter exit (atexit + signal).
    """
    global _llama_proc, _atexit_registered
    config = config or MempalaceConfig()

    if config.embed_provider != "llama-cpp":
        return None

    base_url = os.environ.get("LLAMA_CPP_BASE_URL", "")
    if base_url and _is_server_ready(base_url):
        logger.info("llama-server already running at %s", base_url)
        return base_url

    if _llama_proc is not None and _llama_proc.poll() is None:
        return os.environ.get("LLAMA_CPP_BASE_URL", "http://localhost:8080")

    llama_bin = shutil.which("llama-server")
    if not llama_bin:
        logger.warning("llama-server not found in PATH — install with: brew install llama.cpp")
        return None

    model_path = os.environ.get("LLAMA_CPP_MODEL") or _find_model_path(config)
    if not model_path:
        logger.warning(
            "No bge-m3 GGUF found. Run: ollama pull bge-m3  (to download the model)"
        )
        return None

    if port == 0:
        port = int(os.environ.get("LLAMA_CPP_PORT", "8784"))

    base_url = f"http://localhost:{port}"
    if _is_server_ready(base_url):
        os.environ["LLAMA_CPP_BASE_URL"] = base_url
        logger.info("llama-server already running at %s", base_url)
        return base_url

    logger.info("Starting llama-server on port %d (model: %s)...", port, os.path.basename(model_path))
    _llama_proc = subprocess.Popen(
        [
            llama_bin, "--model", model_path, "--embedding",
            "--port", str(port),
            "--batch-size", "8192", "--ubatch-size", "8192",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _atexit_registered:
        atexit.register(stop_llama_server)
        _atexit_registered = True

    # Wait for server to become ready (max 30s for model load)
    for _ in range(60):
        if _llama_proc.poll() is not None:
            logger.error("llama-server exited with code %d", _llama_proc.returncode)
            _llama_proc = None
            return None
        if _is_server_ready(base_url):
            os.environ["LLAMA_CPP_BASE_URL"] = base_url
            logger.info("llama-server ready at %s (pid %d)", base_url, _llama_proc.pid)
            return base_url
        time.sleep(0.5)

    logger.error("llama-server failed to start within 30s")
    stop_llama_server()
    return None


def stop_llama_server():
    """Kill the managed llama-server subprocess."""
    global _llama_proc
    if _llama_proc is not None and _llama_proc.poll() is None:
        logger.info("Stopping llama-server (pid %d)...", _llama_proc.pid)
        _llama_proc.terminate()
        try:
            _llama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _llama_proc.kill()
    _llama_proc = None
