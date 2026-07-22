"""
Step 5 — Vector Store
Creates a persistent Chroma collection and loads every chunk into it, along
with its embedding and metadata (category, source file, document id) so we
can filter and trace results back to the original article later.
"""

import os

import chromadb
import pandas as pd

import config
from _pipeline_utils import load_module

vector_representation = load_module("04_vector_representation.py", "vector_representation")


def get_chroma_collection(reset=False):
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_store(chunks_df, reset=True, batch_size=2000):
    """
    Adds chunks to Chroma in batches, since Chroma enforces a max batch size
    per .add() call (the exact limit depends on your local SQLite build —
    it showed 5461 in your error, so 2000 stays safely under it).
    """
    collection = get_chroma_collection(reset=reset)
    embeddings = vector_representation.embed_texts(chunks_df["chunk_text"].tolist())

    metadatas = [
        {
            "document_id": int(row["document_id"]),
            "category": row["category"],
            "source_file": row["source_file"],
        }
        for _, row in chunks_df.iterrows()
    ]

    ids = chunks_df["chunk_id"].tolist()
    documents = chunks_df["chunk_text"].tolist()
    embeddings_list = embeddings.tolist()

    total = len(ids)
    for start in range(0, total, batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Added batch {start}-{min(end, total)} of {total}")

    return collection


def main():
    if not os.path.exists(config.CHUNKS_CSV):
        raise FileNotFoundError(
            f"'{config.CHUNKS_CSV}' not found. Run 03_chunking.py first."
        )

    chunks_df = pd.read_csv(config.CHUNKS_CSV)
    collection = build_store(chunks_df, reset=True)

    print(f"Chroma collection '{config.CHROMA_COLLECTION_NAME}' now has {collection.count()} chunks.")
    print(f"Persisted -> {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
