"""
Main entry point for text embedding generation and similarity analysis.

COMPLETE WORKFLOW:
Step 1: Configuration & Initialization (runs automatically on import)
Step 2: Define sample sentences
Step 3: Generate embeddings for each sentence
Step 4: Display embedding properties
Step 5: Calculate semantic similarities between sentence pairs
Step 6: Display results
"""
import sys
import numpy as np
import logging
from generate_embeddings import get_embedding, cosine_similarity, logger

if __name__ == "__main__":
    try:
        # Step 1: Configuration & Initialization (completed on import)
        # This step runs automatically when 'generate_embeddings.py' is imported above.
        # It loads environment variables, validates them, and initializes the OpenAI Embeddings model.

        # Step 2: Define test sentences with varying semantic similarity
        sentences = {
            "tech1": "Artificial intelligence is transforming industries",
            "tech2": "Machine learning revolutionizes business processes",
            "food1": "Pizza is a delicious Italian dish",
            "animal1": "Dogs are loyal and friendly companions"
        }
        logger.info("Step 2: Defined sample sentences.")

        logger.info("=" * 70)
        logger.info("GENERATING EMBEDDINGS")
        logger.info("=" * 70)

        # Step 3: Generate embeddings for all sentences
        embeddings = {}
        logger.info("Step 3: Starting embedding generation...")

        for key, sentence in sentences.items():
            try:
                logger.info(f"\nProcessing: '{sentence}'")
                embedding = get_embedding(sentence)
                embeddings[key] = embedding
                logger.info(f"  ✓ Generated embedding for '{key}'")

                # Step 4: Display embedding properties
                logger.info(f"  → Embedding dimensions: {len(embedding)}")
                logger.info(f"  → First 10 values: {embedding[:10]}...")
                logger.info(f"  → Vector magnitude: {np.linalg.norm(embedding):.4f}")

            except Exception as e:
                logger.error(f"✗ Failed to process sentence '{key}': {str(e)}")
                continue

        if not embeddings:
            logger.error("No embeddings were successfully generated. Cannot proceed with similarity analysis.")
            sys.exit(1)
        logger.info("Step 3 & 4 Complete: All available embeddings generated and inspected.")

        # Step 5: Calculate semantic similarities between sentence pairs
        logger.info("\n" + "=" * 70)
        logger.info("SEMANTIC SIMILARITY ANALYSIS")
        logger.info("=" * 70)

        comparisons = [
            ("tech1", "tech2", "Two tech-related sentences (expected high similarity)"),
            ("tech1", "food1", "Tech vs Food (expected low similarity)"),
            ("tech1", "animal1", "Tech vs Animals (expected low similarity)"),
            ("food1", "animal1", "Food vs Animals (expected medium similarity)")
        ]
        logger.info("Step 5: Starting similarity calculations...")

        for key1, key2, description in comparisons:
            try:
                if key1 not in embeddings or key2 not in embeddings:
                    logger.warning(f"Skipping comparison {key1} vs {key2} - missing embeddings")
                    continue

                similarity = cosine_similarity(embeddings[key1], embeddings[key2])
                logger.info(f"  ✓ Calculated similarity for '{key1}' vs '{key2}'")

                # Step 6: Display similarity results
                logger.info(f"\n{description}:")
                logger.info(f"  '{sentences[key1]}'")
                logger.info("  vs")
                logger.info(f"  '{sentences[key2]}'")
                logger.info(f"  → Similarity: {similarity:.4f}")

            except Exception as e:
                logger.error(f"✗ Failed to calculate similarity between {key1} and {key2}: {str(e)}")
                continue

        logger.info("Step 5 & 6 Complete: All similarity analyses performed and results displayed.")
        logger.info("\n" + "=" * 70)
        logger.info("EMBEDDING ANALYSIS COMPLETE")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        raise

