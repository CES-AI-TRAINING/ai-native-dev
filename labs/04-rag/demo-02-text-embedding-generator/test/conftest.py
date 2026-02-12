"""
Pytest configuration and shared fixtures for embedding tests.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock, Mock

# Set up environment variables before any imports to allow module initialization
os.environ["OPENAI_API_EMBEDDING_KEY"] = "mock_api_key"
os.environ["OPENAI_MODEL"] = "text-embedding-3-small"

# Patch OpenAIEmbeddings class to return a mock instance when initialized
_mock_embeddings_instance = MagicMock()
_patcher = patch('main.generate_embeddings.OpenAIEmbeddings', return_value=_mock_embeddings_instance)


@pytest.fixture(scope="session", autouse=True)
def setup_module_mocks():
    """Set up module-level mocks before any tests run."""
    _patcher.start()
    yield
    _patcher.stop()


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Mock environment variables for all tests to prevent real API calls."""
    monkeypatch.setenv("OPENAI_API_EMBEDDING_KEY", "mock_api_key")
    monkeypatch.setenv("OPENAI_MODEL", "text-embedding-3-small")


@pytest.fixture(autouse=True)
def reset_embeddings_mock():
    """Reset the embeddings mock before each test."""
    _mock_embeddings_instance.reset_mock()
    yield
    _mock_embeddings_instance.reset_mock()


@pytest.fixture
def mock_embeddings_model():
    """Fixture providing access to the mock embeddings model."""
    return _mock_embeddings_instance


@pytest.fixture
def sample_embedding_1536():
    """Fixture providing a sample 1536-dimensional embedding vector."""
    import numpy as np
    np.random.seed(42)
    return np.random.rand(1536).tolist()

