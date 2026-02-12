"""
Text embedding generation module using OpenAI.

This module provides functions to generate text embeddings and calculate
cosine similarity between embedding vectors.

COMPLETE WORKFLOW (when imported by main.py):
Step 1: Configuration & Initialization (runs automatically on import)
Step 2: Text to Vector Conversion (called by main.py via get_embedding())
Step 3: Similarity Calculation (called by main.py via cosine_similarity())
"""
import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: CONFIGURATION & INITIALIZATION
# ============================================================================
# This step runs automatically when this module is imported by main.py
# It sets up the OpenAI connection by:
# - Loading environment variables from .env file
# - Extracting configuration values
# - Validating all required variables are present
# - Initializing OpenAI Embeddings model
# 
# NOTE: This happens before main.py starts executing its steps
# ============================================================================

# Load environment variables from .env file
try:
    load_dotenv()
    logger.info("✓ Environment variables loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load .env file: {e}")
    raise

# Extract configuration values from environment variables
OPENAI_API_EMBEDDING_KEY = os.getenv("OPENAI_API_EMBEDDING_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")  # Default model

# Validate all required environment variables are present
required_vars = {
    "OPENAI_API_EMBEDDING_KEY": OPENAI_API_EMBEDDING_KEY
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file."
    logger.error(f"✗ {error_msg}")
    raise ValueError(error_msg)

logger.info("✓ All required environment variables validated")

# ============================================================================
# INITIALIZE OPENAI EMBEDDINGS MODEL
# ============================================================================
# This creates a reusable model instance that will be used for all embedding calls.
# It is configured with:
# - OpenAI API key
# - OpenAI embedding model (text-embedding-3-small)
# ============================================================================
try:
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_EMBEDDING_KEY,
        model=OPENAI_MODEL
    )
    logger.info(f"✓ OpenAI Embeddings model initialized successfully (model: {OPENAI_MODEL})")
except Exception as e:
    logger.error(f"✗ Failed to initialize OpenAIEmbeddings: {e}")
    raise



# ============================================================================
# STEP 2: TEXT TO VECTOR CONVERSION FUNCTION
# ============================================================================
# This function converts text into embedding vectors through these steps:
# - Validate input text (not empty/None)
# - Call Azure OpenAI API via LangChain to generate embedding
# - Validate the response (ensure we got valid data)
# - Return the embedding vector
# This function is called from 'main.py' (Unified Flow Step 3).
# ============================================================================

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text string.

    This function converts text into a high-dimensional vector representation
    that captures semantic meaning. The embedding can be used for similarity
    comparison with other text embeddings.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding vector (typically 1536 dimensions)

    Raises:
        ValueError: If text is empty or None, or if API returns empty response
        Exception: If API call fails
    """
    try:
        # Validate input text - prevents unnecessary API calls and catches errors early
        if not text or (isinstance(text, str) and not text.strip()):
            raise ValueError("Text cannot be empty or None")

        # ============================================================================
        # GENERATE EMBEDDING FOR TEXT
        # ============================================================================
        # This step uses the initialized 'embeddings_model' to:
        # - Send the text to OpenAI API
        # - Generate a vector embedding (1536 dimensions for text-embedding-3-small)
        # - LangChain handles: HTTP requests, authentication, retries, response parsing
        # ============================================================================
        embedding = embeddings_model.embed_query(text)

        # Validate the response - ensure we received valid data from the API
        if not embedding or len(embedding) == 0:
            raise ValueError("Empty embedding received from API")

        # Return the embedding vector
        return embedding

    except Exception as e:
        # Error handling: Log error safely without exposing full text
        text_preview = str(text)[:50] if text else "None or empty"
        logger.error(f"✗ Failed to get embedding for text '{text_preview}...': {str(e)}")
        raise


# ============================================================================
# STEP 3: SIMILARITY CALCULATION FUNCTION
# ============================================================================
# This function calculates cosine similarity between two vectors through these steps:
# - Validate both vectors (not None, not empty, same dimensions)
# - Convert lists to NumPy arrays for efficient computation
# - Calculate dot product (A · B)
# - Calculate magnitudes (||A|| and ||B||)
# - Check for zero magnitude (edge case handling)
# - Apply formula: similarity = (A · B) / (||A|| × ||B||)
# This function is called from 'main.py' (Unified Flow Step 5).
# ============================================================================

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    This function implements the cosine similarity formula:
        similarity = (A · B) / (||A|| × ||B||)
    
    Where:
    - A · B is the dot product (sum of element-wise products)
    - ||A|| and ||B|| are the magnitudes (L2 norms = Euclidean distance from origin)
    
    The result measures the cosine of the angle between two vectors:
    - 1.0 = identical direction (most similar)
    - 0.0 = orthogonal (unrelated)
    - -1.0 = opposite direction (least similar)
    
    For text embeddings, values typically range from 0.0 to 1.0 (rarely negative).

    Args:
        vec1: First vector as list of floats
        vec2: Second vector as list of floats

    Returns:
        Float between -1 and 1 (higher = more similar)

    Raises:
        ValueError: If vectors are empty, None, or have different dimensions
        ZeroDivisionError: If either vector has zero magnitude
    """
    try:
        # Validate both vectors - check they're not None, not empty, and have same dimensions
        if not vec1 or not vec2:
            raise ValueError("Vectors cannot be None or empty")

        if len(vec1) != len(vec2):
            raise ValueError(f"Vectors must have same dimensions. Got {len(vec1)} and {len(vec2)}")

        if len(vec1) == 0:
            raise ValueError("Vectors cannot be empty")

        # Convert lists to NumPy arrays - provides efficient mathematical operations for large vectors
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)

        # Calculate dot product (A · B) - sum of element-wise products: Σ(A[i] × B[i])
        dot_product = np.dot(vec1_array, vec2_array)
        
        # Calculate magnitudes (||A|| and ||B||) - L2 norm = Euclidean distance from origin = √(Σ(x²))
        magnitude1 = np.linalg.norm(vec1_array)
        magnitude2 = np.linalg.norm(vec2_array)

        # Check for zero magnitude (edge case) - prevents division by zero error
        if magnitude1 == 0 or magnitude2 == 0:
            raise ZeroDivisionError("Cannot calculate cosine similarity with zero-magnitude vector")

        # Apply cosine similarity formula - final calculation: (A · B) / (||A|| × ||B||)
        similarity = float(dot_product / (magnitude1 * magnitude2))
        return similarity

    except Exception as e:
        # Log unexpected errors
        logger.error(f"✗ Unexpected error calculating cosine similarity: {str(e)}")
        raise