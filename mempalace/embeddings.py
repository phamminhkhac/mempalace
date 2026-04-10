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
        kwargs["metadata"] = {"hnsw:space": "cosine"}
        return client.get_or_create_collection(**kwargs)
    return client.get_collection(**kwargs)


# ---------------------------------------------------------------------------
# llama-server process manager — shared across MCP instances via ref-counting
#
# State directory: ~/.mempalace/llama-server/
#   pid        — PID of the shared llama-server process
#   refs/      — one file per MCP client process (filename = client PID)
#
# Start: register ref, spawn server if not already running
# Stop:  unregister ref, kill server only when no refs remain
# ---------------------------------------------------------------------------

_LLAMA_DIR = os.path.expanduser("~/.mempalace/llama-server")
_LLAMA_PID_FILE = os.path.join(_LLAMA_DIR, "pid")
_LLAMA_REFS_DIR = os.path.join(_LLAMA_DIR, "refs")
_atexit_registered: bool = False


def _find_model_path() -> Optional[str]:
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


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_server_pid() -> Optional[int]:
    """Read the shared server PID from disk, return None if stale or missing."""
    try:
        with open(_LLAMA_PID_FILE) as f:
            pid = int(f.read().strip())
        if _pid_alive(pid):
            return pid
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def _register_ref():
    """Register this process as a client of the shared llama-server."""
    os.makedirs(_LLAMA_REFS_DIR, exist_ok=True)
    ref_file = os.path.join(_LLAMA_REFS_DIR, str(os.getpid()))
    with open(ref_file, "w") as f:
        f.write("")


def _unregister_ref():
    """Unregister this process. Returns number of remaining live refs."""
    my_ref = os.path.join(_LLAMA_REFS_DIR, str(os.getpid()))
    try:
        os.unlink(my_ref)
    except FileNotFoundError:
        pass
    # Count remaining refs (only those with alive PIDs)
    remaining = 0
    try:
        for name in os.listdir(_LLAMA_REFS_DIR):
            try:
                pid = int(name)
                if _pid_alive(pid):
                    remaining += 1
                else:
                    # Clean up stale ref
                    os.unlink(os.path.join(_LLAMA_REFS_DIR, name))
            except (ValueError, OSError):
                pass
    except FileNotFoundError:
        pass
    return remaining


def start_llama_server(config: Optional[MempalaceConfig] = None, port: int = 0) -> Optional[str]:
    """Start or attach to a shared llama-server.

    Multiple MCP processes share one server via ref-counting.
    The server is only started if no existing instance is running.
    """
    global _atexit_registered
    config = config or MempalaceConfig()

    if config.embed_provider != "llama-cpp":
        return None

    if port == 0:
        port = int(os.environ.get("LLAMA_CPP_PORT", "8784"))
    base_url = os.environ.get("LLAMA_CPP_BASE_URL", f"http://localhost:{port}")

    # Register this process as a client
    _register_ref()
    if not _atexit_registered:
        atexit.register(stop_llama_server)
        _atexit_registered = True

    # Check if server is already running (started by another MCP or manually)
    if _is_server_ready(base_url):
        os.environ["LLAMA_CPP_BASE_URL"] = base_url
        logger.info("llama-server already running at %s", base_url)
        return base_url

    # Check PID file for a process we previously started
    existing_pid = _read_server_pid()
    if existing_pid and _is_server_ready(base_url):
        os.environ["LLAMA_CPP_BASE_URL"] = base_url
        logger.info("llama-server already running at %s (pid %d)", base_url, existing_pid)
        return base_url

    # Need to start a new server
    llama_bin = shutil.which("llama-server")
    if not llama_bin:
        logger.warning("llama-server not found in PATH — install with: brew install llama.cpp")
        return None

    model_path = os.environ.get("LLAMA_CPP_MODEL") or _find_model_path()
    if not model_path:
        logger.warning("No bge-m3 GGUF found. Run: ollama pull bge-m3")
        return None

    logger.info("Starting llama-server on port %d (model: %s)...", port, os.path.basename(model_path))
    proc = subprocess.Popen(
        [
            llama_bin, "--model", model_path, "--embedding",
            "--port", str(port),
            "--batch-size", "8192", "--ubatch-size", "8192",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write PID file
    os.makedirs(_LLAMA_DIR, exist_ok=True)
    with open(_LLAMA_PID_FILE, "w") as f:
        f.write(str(proc.pid))

    # Wait for server to become ready (max 30s)
    for _ in range(60):
        if proc.poll() is not None:
            logger.error("llama-server exited with code %d", proc.returncode)
            return None
        if _is_server_ready(base_url):
            os.environ["LLAMA_CPP_BASE_URL"] = base_url
            logger.info("llama-server ready at %s (pid %d)", base_url, proc.pid)
            return base_url
        time.sleep(0.5)

    logger.error("llama-server failed to start within 30s")
    proc.kill()
    return None


def stop_llama_server():
    """Unregister this client. Kill the server only when no clients remain."""
    remaining = _unregister_ref()
    if remaining > 0:
        logger.info("Unregistered from llama-server (%d clients still active)", remaining)
        return

    # No clients left — kill the server
    pid = _read_server_pid()
    if pid:
        logger.info("No clients left, stopping llama-server (pid %d)...", pid)
        try:
            os.kill(pid, 15)  # SIGTERM
        except OSError:
            pass
    # Clean up state
    try:
        os.unlink(_LLAMA_PID_FILE)
    except FileNotFoundError:
        pass
