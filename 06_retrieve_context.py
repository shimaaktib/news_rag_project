"""
Step 6 — Context Retrieval
Given a user question, embeds it, queries the Chroma vector store for the
closest chunks, then assembles a clean "context package": score-filtered,
de-duplicated, capped per document, and kept inside a word budget so a single
long article can't crowd out everything else.
"""

import os

import config
from _pipeline_utils import load_module

vector_representation = load_module("04_vector_representation.py", "vector_representation")
chroma_store = load_module("05_create_chroma_store.py", "chroma_store")


def retrieve_top_k(query, k=config.RETRIEVAL_K):
    collection = chroma_store.get_chroma_collection(reset=False)
    query_embedding = vector_representation.embed_texts([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        candidates.append({
            "chunk_id": results["ids"][0][i],
            "chunk_text": results["documents"][0][i],
            "document_id": results["metadatas"][0][i]["document_id"],
            "category": results["metadatas"][0][i]["category"],
            "source_file": results["metadatas"][0][i]["source_file"],
            # cosine distance -> similarity score in [0, 1] (higher is better)
            "score": 1 - distance,
        })
    return candidates


def build_context_package(
    query,
    retrieval_k=config.RETRIEVAL_K,
    max_context_chunks=config.MAX_CONTEXT_CHUNKS,
    max_chunks_per_document=config.MAX_CHUNKS_PER_DOCUMENT,
    word_budget=config.WORD_BUDGET,
    min_score_ratio=0.30,
    min_absolute_score=0.05,
):
    candidates = retrieve_top_k(query, k=retrieval_k)
    candidates.sort(key=lambda row: row["score"], reverse=True)

    max_score = candidates[0]["score"] if candidates else 0.0
    selected_rows = []
    seen_texts = set()
    per_document_counts = {}
    used_words = 0

    for row in candidates:
        if row["score"] < min_absolute_score:
            continue
        if max_score > 0 and row["score"] < max_score * min_score_ratio:
            continue

        normalized = " ".join(row["chunk_text"].split()).strip().lower()
        if normalized in seen_texts:
            continue

        doc_count = per_document_counts.get(row["document_id"], 0)
        if doc_count >= max_chunks_per_document:
            continue

        chunk_words = len(row["chunk_text"].split())
        if selected_rows and used_words + chunk_words > word_budget:
            continue

        selected_rows.append(row)
        seen_texts.add(normalized)
        per_document_counts[row["document_id"]] = doc_count + 1
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    blocks = []
    for position, row in enumerate(selected_rows, start=1):
        blocks.append(
            f"[Source {position}] category: {row['category']} | file: {row['source_file']}\n"
            f"{row['chunk_text']}"
        )

    return {
        "query": query,
        "candidates": candidates,
        "selected": selected_rows,
        "context_text": "\n\n".join(blocks),
        "used_words": used_words,
        "num_sources": len(selected_rows),
    }


def main():
    sample_query = "What increased profits for the company this quarter?"
    package = build_context_package(sample_query)

    print(f"Query: {sample_query}")
    print(f"Sources selected: {package['num_sources']} | words used: {package['used_words']}")
    print()
    print(package["context_text"] or "(no context passed the score/word-budget filters)")


if __name__ == "__main__":
    if not os.path.isdir(config.CHROMA_DIR) or not os.listdir(config.CHROMA_DIR):
        raise FileNotFoundError(
            f"'{config.CHROMA_DIR}' is empty. Run 05_create_chroma_store.py first."
        )
    main()
