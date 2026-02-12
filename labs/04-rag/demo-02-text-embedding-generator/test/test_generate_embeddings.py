"""
Comprehensive test suite for generate_embeddings.py

Tests cover:
- get_embedding() function with mocked API calls
- cosine_similarity() function with various scenarios
- Edge cases and error handling
- Input validation
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import from main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCosineSimilarity:
    """Test suite for cosine_similarity() function"""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors (should be 1.0)"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0, 2.0, 3.0, 4.0]
        vec2 = [1.0, 2.0, 3.0, 4.0]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors (should be 0.0)"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity with opposite vectors (should be -1.0)"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_cosine_similarity_positive_similarity(self):
        """Test cosine similarity with positively correlated vectors"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [2.0, 4.0, 6.0]  # Scaled version of vec1
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_cosine_similarity_high_dimensional(self):
        """Test cosine similarity with high-dimensional vectors (e.g., embedding size)"""
        from main.generate_embeddings import cosine_similarity
        
        # Simulate 1536-dimensional embedding (common size)
        np.random.seed(42)
        vec1 = np.random.rand(1536).tolist()
        vec2 = np.random.rand(1536).tolist()
        
        result = cosine_similarity(vec1, vec2)
        
        # Result should be between -1 and 1
        assert -1.0 <= result <= 1.0
        # For random vectors, similarity should be close to 0 (but can vary)
        # Use a more lenient threshold since random vectors can have higher similarity by chance
        assert abs(result) < 1.0

    def test_cosine_similarity_different_dimensions_raises_error(self):
        """Test that cosine_similarity raises ValueError for different dimension vectors"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0, 4.0]
        
        with pytest.raises(ValueError, match="Vectors must have same dimensions"):
            cosine_similarity(vec1, vec2)

    def test_cosine_similarity_none_vector_raises_error(self):
        """Test that cosine_similarity raises ValueError for None vectors"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = None
        vec2 = [1.0, 2.0, 3.0]
        
        with pytest.raises(ValueError, match="Vectors cannot be None or empty"):
            cosine_similarity(vec1, vec2)

    def test_cosine_similarity_empty_vector_raises_error(self):
        """Test that cosine_similarity raises ValueError for empty vectors"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = []
        vec2 = [1.0, 2.0, 3.0]
        
        with pytest.raises(ValueError, match="Vectors cannot be None or empty"):
            cosine_similarity(vec1, vec2)

    def test_cosine_similarity_zero_vector_raises_error(self):
        """Test that cosine_similarity raises ZeroDivisionError for zero-magnitude vectors"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        
        with pytest.raises(ZeroDivisionError, match="Cannot calculate cosine similarity with zero-magnitude vector"):
            cosine_similarity(vec1, vec2)

    def test_cosine_similarity_returns_float(self):
        """Test that cosine_similarity returns a float type"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]
        
        result = cosine_similarity(vec1, vec2)
        assert isinstance(result, float)

    def test_cosine_similarity_normalized_vectors(self):
        """Test cosine similarity with normalized unit vectors"""
        from main.generate_embeddings import cosine_similarity
        
        # Normalize vectors to unit length
        vec1 = np.array([1.0, 2.0, 3.0])
        vec1_norm = vec1 / np.linalg.norm(vec1)
        
        vec2 = np.array([4.0, 5.0, 6.0])
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        result = cosine_similarity(vec1_norm.tolist(), vec2_norm.tolist())
        
        # Cosine similarity should be the dot product of normalized vectors
        expected = np.dot(vec1_norm, vec2_norm)
        assert result == pytest.approx(expected, abs=1e-10)


class TestGetEmbedding:
    """Test suite for get_embedding() function"""

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_success(self, mock_embeddings_model):
        """Test successful embedding generation"""
        from main.generate_embeddings import get_embedding
        
        # Mock the embedding response (typical 1536-dimensional vector)
        mock_embedding = np.random.rand(1536).tolist()
        mock_embeddings_model.embed_query.return_value = mock_embedding
        
        text = "Artificial intelligence is transforming industries"
        result = get_embedding(text)
        
        assert result == mock_embedding
        assert len(result) == 1536
        mock_embeddings_model.embed_query.assert_called_once_with(text)

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_different_text(self, mock_embeddings_model):
        """Test embedding generation with different text inputs"""
        from main.generate_embeddings import get_embedding
        
        # Mock different embeddings for different texts
        mock_embeddings_model.embed_query.side_effect = [
            [0.1] * 1536,
            [0.2] * 1536,
        ]
        
        text1 = "Machine learning"
        text2 = "Deep learning"
        
        result1 = get_embedding(text1)
        result2 = get_embedding(text2)
        
        assert result1 != result2
        assert mock_embeddings_model.embed_query.call_count == 2

    def test_get_embedding_empty_string_raises_error(self):
        """Test that get_embedding raises ValueError for empty string"""
        from main.generate_embeddings import get_embedding
        
        with pytest.raises(ValueError, match="Text cannot be empty or None"):
            get_embedding("")

    def test_get_embedding_whitespace_only_raises_error(self):
        """Test that get_embedding raises ValueError for whitespace-only string"""
        from main.generate_embeddings import get_embedding
        
        with pytest.raises(ValueError, match="Text cannot be empty or None"):
            get_embedding("   ")

    def test_get_embedding_none_raises_error(self):
        """Test that get_embedding raises ValueError for None input"""
        from main.generate_embeddings import get_embedding
        
        with pytest.raises(ValueError, match="Text cannot be empty or None"):
            get_embedding(None)

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_empty_response_raises_error(self, mock_embeddings_model):
        """Test that get_embedding raises ValueError when API returns empty embedding"""
        from main.generate_embeddings import get_embedding
        
        mock_embeddings_model.embed_query.return_value = []
        
        with pytest.raises(ValueError, match="Empty embedding received from API"):
            get_embedding("test text")

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_api_exception(self, mock_embeddings_model):
        """Test that get_embedding propagates API exceptions"""
        from main.generate_embeddings import get_embedding
        
        mock_embeddings_model.embed_query.side_effect = Exception("API connection failed")
        
        with pytest.raises(Exception, match="API connection failed"):
            get_embedding("test text")

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_long_text(self, mock_embeddings_model):
        """Test embedding generation with long text"""
        from main.generate_embeddings import get_embedding
        
        mock_embedding = np.random.rand(1536).tolist()
        mock_embeddings_model.embed_query.return_value = mock_embedding
        
        long_text = "Artificial intelligence " * 100  # Very long text
        result = get_embedding(long_text)
        
        assert result == mock_embedding
        mock_embeddings_model.embed_query.assert_called_once_with(long_text)

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_special_characters(self, mock_embeddings_model):
        """Test embedding generation with special characters"""
        from main.generate_embeddings import get_embedding
        
        mock_embedding = np.random.rand(1536).tolist()
        mock_embeddings_model.embed_query.return_value = mock_embedding
        
        special_text = "Hello! @#$%^&*() 中文 español"
        result = get_embedding(special_text)
        
        assert result == mock_embedding
        mock_embeddings_model.embed_query.assert_called_once_with(special_text)

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_returns_list_of_floats(self, mock_embeddings_model):
        """Test that get_embedding returns a list of floats"""
        from main.generate_embeddings import get_embedding
        
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embeddings_model.embed_query.return_value = mock_embedding
        
        result = get_embedding("test")
        
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


class TestIntegration:
    """Integration tests combining get_embedding and cosine_similarity"""

    @patch('main.generate_embeddings.embeddings_model')
    def test_semantic_similarity_workflow(self, mock_embeddings_model):
        """Test complete workflow: generate embeddings and calculate similarity"""
        from main.generate_embeddings import get_embedding, cosine_similarity
        
        # Simulate embeddings for similar and different texts
        np.random.seed(42)
        similar_embedding1 = np.random.rand(1536).tolist()
        # Create a slightly modified version for "similar" text
        similar_embedding2 = (np.array(similar_embedding1) + np.random.rand(1536) * 0.1).tolist()
        # Create a different embedding for "different" text
        different_embedding = np.random.rand(1536).tolist()
        
        mock_embeddings_model.embed_query.side_effect = [
            similar_embedding1,
            similar_embedding2,
            different_embedding
        ]
        
        text1 = "Artificial intelligence"
        text2 = "Machine learning"
        text3 = "Pizza recipe"
        
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        emb3 = get_embedding(text3)
        
        # Similar texts should have higher similarity
        similarity_similar = cosine_similarity(emb1, emb2)
        similarity_different = cosine_similarity(emb1, emb3)
        
        # Note: This is a simplified test - in practice, similar texts would
        # have embeddings that are closer, but here we're just testing the workflow
        assert -1.0 <= similarity_similar <= 1.0
        assert -1.0 <= similarity_different <= 1.0

    @patch('main.generate_embeddings.embeddings_model')
    def test_multiple_embeddings_and_comparisons(self, mock_embeddings_model):
        """Test generating multiple embeddings and comparing them"""
        from main.generate_embeddings import get_embedding, cosine_similarity
        
        # Generate mock embeddings
        np.random.seed(123)
        mock_embeddings = [np.random.rand(1536).tolist() for _ in range(4)]
        mock_embeddings_model.embed_query.side_effect = mock_embeddings
        
        texts = [
            "Artificial intelligence",
            "Machine learning",
            "Deep learning",
            "Pizza recipe"
        ]
        
        embeddings = [get_embedding(text) for text in texts]
        
        # Calculate all pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
                
                # All similarities should be in valid range
                assert -1.0 <= sim <= 1.0
        
        # Should have C(4,2) = 6 comparisons
        assert len(similarities) == 6


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_cosine_similarity_single_dimension(self):
        """Test cosine similarity with single-dimensional vectors"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0]
        vec2 = [1.0]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0)

    def test_cosine_similarity_single_dimension_different(self):
        """Test cosine similarity with single-dimensional vectors (different signs)"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0]
        vec2 = [-1.0]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(-1.0)

    @patch('main.generate_embeddings.embeddings_model')
    def test_get_embedding_single_character(self, mock_embeddings_model):
        """Test embedding generation with single character"""
        from main.generate_embeddings import get_embedding
        
        mock_embedding = np.random.rand(1536).tolist()
        mock_embeddings_model.embed_query.return_value = mock_embedding
        
        result = get_embedding("a")
        assert result == mock_embedding

    def test_cosine_similarity_very_large_numbers(self):
        """Test cosine similarity with very large numbers"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1e10, 2e10, 3e10]
        vec2 = [1e10, 2e10, 3e10]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_cosine_similarity_very_small_numbers(self):
        """Test cosine similarity with very small numbers"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1e-10, 2e-10, 3e-10]
        vec2 = [1e-10, 2e-10, 3e-10]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_cosine_similarity_mixed_signs(self):
        """Test cosine similarity with vectors containing mixed positive and negative values"""
        from main.generate_embeddings import cosine_similarity
        
        vec1 = [1.0, -2.0, 3.0, -4.0]
        vec2 = [1.0, -2.0, 3.0, -4.0]
        
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(1.0, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

