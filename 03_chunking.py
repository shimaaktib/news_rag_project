"""
Step 3 — Chunking
Splits each (preprocessed) article into overlapping, fixed-size chunks so the
retriever works on manageable pieces of text instead of whole articles.
"""

import os

import pandas as pd

import config


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(preprocessed_df):
    chunk_rows = []
    for _, doc in preprocessed_df.iterrows():
        for chunk_index, chunk in enumerate(chunk_text(doc["clean_text"])):
            chunk_rows.append({
                "chunk_id": f"doc{doc['document_id']}_chunk{chunk_index}",
                "document_id": doc["document_id"],
                "category": doc["category"],
                "source_file": doc["filename"],
                "chunk_index": chunk_index,
                "chunk_text": chunk,
            })
    return pd.DataFrame(chunk_rows)


def main():
    if not os.path.exists(config.PREPROCESSED_CSV):
        raise FileNotFoundError(
            f"'{config.PREPROCESSED_CSV}' not found. Run 02_preprocessing.py first."
        )

    preprocessed_df = pd.read_csv(config.PREPROCESSED_CSV)
    chunks_df = build_chunks(preprocessed_df)

    chunks_df.to_csv(config.CHUNKS_CSV, index=False)
    print(f"Created {len(chunks_df)} chunks from {len(preprocessed_df)} documents.")
    print(chunks_df.head(5))
    print(f"Saved -> {config.CHUNKS_CSV}")


if __name__ == "__main__":
    main()
