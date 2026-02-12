# Test Suite for generate_embeddings.py

This directory contains comprehensive test cases for the embedding generation module.

## Test Structure

- `test_generate_embeddings.py` - Main test suite covering:
  - `cosine_similarity()` function tests
  - `get_embedding()` function tests  
  - Integration tests
  - Edge cases and error handling

- `conftest.py` - Pytest configuration with shared fixtures and mocks
- `pytest.ini` - Pytest configuration file

## Running Tests

### Install Dependencies

First, install test dependencies:

```bash
uv sync
```

### Run All Tests

```bash
# Using uv
uv run pytest

# Using pytest directly (if installed globally)
pytest
```

### Run with Coverage

```bash
uv run pytest --cov=generate_embeddings --cov-report=html --cov-report=term
```

### Run Specific Test Classes

```bash
# Run only cosine similarity tests
uv run pytest test/test_generate_embeddings.py::TestCosineSimilarity -v

# Run only embedding generation tests
uv run pytest test/test_generate_embeddings.py::TestGetEmbedding -v

# Run integration tests
uv run pytest test/test_generate_embeddings.py::TestIntegration -v
```

### Run Specific Test Functions

```bash
uv run pytest test/test_generate_embeddings.py::TestCosineSimilarity::test_cosine_similarity_identical_vectors -v
```

## Test Coverage

The test suite covers:

### Cosine Similarity Tests
- ✅ Identical vectors (similarity = 1.0)
- ✅ Orthogonal vectors (similarity = 0.0)
- ✅ Opposite vectors (similarity = -1.0)
- ✅ High-dimensional vectors (1536-dim embeddings)
- ✅ Error handling (empty, None, different dimensions, zero vectors)
- ✅ Edge cases (single dimension, very large/small numbers, mixed signs)

### Get Embedding Tests
- ✅ Successful embedding generation
- ✅ Different text inputs
- ✅ Input validation (empty, whitespace, None)
- ✅ API error handling (empty response, exceptions)
- ✅ Special cases (long text, special characters)

### Integration Tests
- ✅ Complete workflow (embedding + similarity calculation)
- ✅ Multiple embeddings and comparisons

## Mocking Strategy

Tests use mocking to avoid making real API calls:
- Environment variables are mocked via `conftest.py`
- `AzureOpenAIEmbeddings` class is patched to return a mock instance
- `embed_query()` method behavior is configured per test

## Notes

- All tests use mocked API calls - no real Azure OpenAI credentials needed
- Tests are isolated and can run in any order
- Fixtures in `conftest.py` ensure consistent test setup

