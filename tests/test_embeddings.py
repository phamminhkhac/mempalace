"""Tests for mempalace.embeddings — embedding factory and collection helpers."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from mempalace.config import MempalaceConfig
from mempalace.embeddings import (
    GeminiEmbeddingFunction,
    OllamaEmbeddingFunction,
    get_collection,
    get_embedding_function,
)


def _make_config(tmp_dir, **overrides):
    cfg_dir = os.path.join(tmp_dir, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    data = {"palace_path": os.path.join(tmp_dir, "palace")}
    data.update(overrides)
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(data, f)
    return MempalaceConfig(config_dir=cfg_dir)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestGetEmbeddingFunction:
    def test_default_returns_none(self, tmp_dir):
        cfg = _make_config(tmp_dir, embed_provider="default")
        assert get_embedding_function(cfg) is None

    def test_ollama_returns_instance(self, tmp_dir):
        cfg = _make_config(tmp_dir, embed_provider="ollama")
        ef = get_embedding_function(cfg)
        assert isinstance(ef, OllamaEmbeddingFunction)

    def test_ollama_custom_model(self, tmp_dir):
        cfg = _make_config(tmp_dir, embed_provider="ollama", embed_model="nomic-embed-text")
        ef = get_embedding_function(cfg)
        assert ef._model == "nomic-embed-text"

    def test_gemini_missing_key_raises(self, tmp_dir):
        cfg = _make_config(tmp_dir, embed_provider="gemini")
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(ValueError, match="API key"):
                get_embedding_function(cfg)

    def test_env_var_override(self, tmp_dir):
        cfg = _make_config(tmp_dir)  # default in file
        with patch.dict(os.environ, {"MEMPALACE_EMBED_PROVIDER": "ollama"}):
            ef = get_embedding_function(MempalaceConfig(config_dir=os.path.join(tmp_dir, "cfg")))
            assert isinstance(ef, OllamaEmbeddingFunction)


# ---------------------------------------------------------------------------
# OllamaEmbeddingFunction tests
# ---------------------------------------------------------------------------


class TestOllamaEmbedding:
    def test_call_sends_http_request(self):
        ef = OllamaEmbeddingFunction(model="bge-m3")
        fake_response = json.dumps({"embeddings": [[0.1, 0.2], [0.3, 0.4]]}).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = ef(["hello", "world"])
            assert len(result) == 2
            assert list(result[0]) == [0.1, 0.2]
            assert list(result[1]) == [0.3, 0.4]
            # Verify the request was made to the correct URL
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            assert "/api/embed" in req.full_url

    def test_connection_error(self):
        ef = OllamaEmbeddingFunction(model="bge-m3", base_url="http://localhost:99999")
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            with pytest.raises(Exception):
                ef(["hello"])


# ---------------------------------------------------------------------------
# GeminiEmbeddingFunction tests
# ---------------------------------------------------------------------------


class TestGeminiEmbedding:
    def test_import_error_without_sdk(self):
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="google-genai"):
                GeminiEmbeddingFunction(api_key="test-key")


# ---------------------------------------------------------------------------
# Collection helper tests
# ---------------------------------------------------------------------------


class TestGetCollection:
    def test_create_collection(self, palace_path):
        import chromadb

        client = chromadb.PersistentClient(path=palace_path)
        col = get_collection(client, "test_col", create=True)
        assert col.name == "test_col"

    def test_get_nonexistent_raises(self, palace_path):
        import chromadb

        client = chromadb.PersistentClient(path=palace_path)
        with pytest.raises(Exception):
            get_collection(client, "nonexistent_col")
