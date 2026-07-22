"""
Step 4 — Vector Representation
Turns chunk text into dense embedding vectors using a sentence-transformers
model. This step is kept separate from the vector store (step 5) so the
"text -> vector" idea is visible on its own, the way it was in the lab
sequence.
"""

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def main():
    if not os.path.exists(config.CHUNKS_CSV):
        raise FileNotFoundError(
            f"'{config.CHUNKS_CSV}' not found. Run 03_chunking.py first."
        )

    chunks_df = pd.read_csv(config.CHUNKS_CSV)
    embeddings = embed_texts(chunks_df["chunk_text"].tolist())

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    np.save(config.EMBEDDINGS_NPY, embeddings)

    print(f"Embedded {len(chunks_df)} chunks -> shape {embeddings.shape}")
    print(f"Saved -> {config.EMBEDDINGS_NPY}")


if __name__ == "__main__":
    main()
