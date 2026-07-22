"""
Step 6 - Context Retrieval  (Senior-RAG-Engineer upgrade)

Pipeline for a single query:

    1. HYBRID SEARCH      dense (Chroma) + lexical (BM25), fused with
                           Reciprocal Rank Fusion (RRF)
    2. CROSS-ENCODER       re-rank the fused candidates with a real
       RE-RANKING          query/passage relevance model
    3. MMR                 drop near-duplicate chunks from the re-ranked list
    4. DOCUMENT VOTING     aggregate scores per source article; if one
                           article clearly wins, use ONLY that article
                           (prevents cross-document contamination, e.g. two
                           BALCO articles from different dates being merged)
    5. CONTEXT GROUPING    chunks from the same article are merged into a
                           single block instead of being interleaved
    6. DIAGNOSTICS         every candidate keeps its dense/BM25/RRF/rerank
                           scores so retrieval quality can be inspected

Every new behaviour is controlled by flags in config.py
(USE_HYBRID, USE_RERANKER, USE_MMR, CONFIDENCE_GAP, ...) so it can be
switched off without touching this file.
"""

import os
import re

import config
from _pipeline_utils import load_module

vector_representation = load_module("04_vector_representation.py", "vector_representation")
chroma_store = load_module("05_create_chroma_store.py", "chroma_store")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


# ---------------------------------------------------------------------------
# BM25 corpus (lexical index) - built once from whatever is already in Chroma
# ---------------------------------------------------------------------------
def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(collection):
    """
    Pulls every chunk already stored in Chroma and builds a BM25 index over
    it. Kept separate from Chroma itself (Chroma has no native BM25) but
    reuses Chroma as the single source of truth for chunk text/metadata, so
    there's no second copy of the data to keep in sync.
    """
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw["ids"]
    documents = raw["documents"]
    metadatas = raw["metadatas"]

    bm25 = BM25Okapi([_tokenize(doc) for doc in documents]) if BM25Okapi else None
    id_to_index = {chunk_id: i for i, chunk_id in enumerate(ids)}

    return {
        "bm25": bm25,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "id_to_index": id_to_index,
    }


# ---------------------------------------------------------------------------
# Stage 1 - hybrid search + Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _dense_ranked_ids(query_embedding, collection, k):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["distances"],
    )
    ranked = []
    scores = {}
    for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
        score = 1 - distance  # cosine distance -> similarity
        if score >= config.SIMILARITY_THRESHOLD:
            ranked.append(chunk_id)
            scores[chunk_id] = score
    return ranked, scores


def _bm25_ranked_ids(query, bm25_bundle, k):
    if bm25_bundle["bm25"] is None:
        return [], {}
    scores_all = bm25_bundle["bm25"].get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores_all)), key=lambda i: scores_all[i], reverse=True)[:k]
    ranked_ids = [bm25_bundle["ids"][i] for i in ranked_idx]
    scores = {bm25_bundle["ids"][i]: float(scores_all[i]) for i in ranked_idx}
    return ranked_ids, scores


def _reciprocal_rank_fusion(ranked_lists, rrf_k=config.RRF_K):
    """ranked_lists: list of [chunk_id, ...] each already sorted best-first."""
    fused = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return fused  # {chunk_id: rrf_score}


def hybrid_search(query, collection, bm25_bundle, k=config.TOP_K_INITIAL):
    query_embedding = vector_representation.embed_texts([query])[0].tolist()
    dense_ids, dense_scores = _dense_ranked_ids(query_embedding, collection, k)

    if config.USE_HYBRID and bm25_bundle is not None:
        bm25_ids, bm25_scores = _bm25_ranked_ids(query, bm25_bundle, k)
        fused = _reciprocal_rank_fusion([dense_ids, bm25_ids])
    else:
        bm25_scores = {}
        fused = _reciprocal_rank_fusion([dense_ids])

    ranked_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)[:k]

    id_to_index = bm25_bundle["id_to_index"] if bm25_bundle else {}
    candidates = []
    for chunk_id in ranked_ids:
        idx = id_to_index.get(chunk_id)
        if idx is None:
            continue
        meta = bm25_bundle["metadatas"][idx]
        candidates.append({
            "chunk_id": chunk_id,
            "chunk_text": bm25_bundle["documents"][idx],
            "document_id": meta["document_id"],
            "category": meta["category"],
            "source_file": meta["source_file"],
            "dense_score": dense_scores.get(chunk_id),
            "bm25_score": bm25_scores.get(chunk_id),
            "rrf_score": fused[chunk_id],
            "rerank_score": None,
        })
    return candidates


# ---------------------------------------------------------------------------
# Stage 2 - cross-encoder re-ranking
# ---------------------------------------------------------------------------
def rerank_candidates(query, candidates, reranker, top_k=config.TOP_K_RERANK):
    if reranker is None or not candidates:
        return candidates[:top_k]

    pairs = [(query, c["chunk_text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    candidates = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Stage 3 - Maximal Marginal Relevance (drop near-duplicate chunks)
# ---------------------------------------------------------------------------
def apply_mmr(candidates, lambda_mult=config.MMR_LAMBDA):
    if len(candidates) <= 1:
        return candidates

    texts = [c["chunk_text"] for c in candidates]
    embeddings = vector_representation.embed_texts(texts)

    def relevance(c):
        return c.get("rerank_score") if c.get("rerank_score") is not None else c.get("rrf_score", 0.0)

    remaining = list(range(len(candidates)))
    selected = []

    first = max(remaining, key=lambda i: relevance(candidates[i]))
    selected.append(first)
    remaining.remove(first)

    while remaining:
        best_i, best_score = None, float("-inf")
        for i in remaining:
            max_sim = max(float(embeddings[i] @ embeddings[j]) for j in selected)
            mmr_score = lambda_mult * relevance(candidates[i]) - (1 - lambda_mult) * max_sim
            if mmr_score > best_score:
                best_score, best_i = mmr_score, i
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


# ---------------------------------------------------------------------------
# Stage 4+5 - document-level voting & context grouping
# ---------------------------------------------------------------------------
def _candidate_score(c):
    if c.get("rerank_score") is not None:
        return c["rerank_score"]
    return c.get("rrf_score", 0.0)


def group_by_document_with_voting(
    candidates,
    max_documents=config.MAX_CONTEXT_DOCUMENTS,
    max_chunks_per_document=config.MAX_CHUNKS_PER_DOCUMENT,
    confidence_gap=config.CONFIDENCE_GAP,
):
    """
    Aggregates candidate scores per source article. If the best article's
    score clearly beats the runner-up (by `confidence_gap`), only that
    article is kept - this is what stops the LLM from merging facts from
    two different articles about the same event.
    """
    doc_scores = {}
    doc_candidates = {}
    for c in candidates:
        doc_id = c["document_id"]
        doc_scores[doc_id] = max(doc_scores.get(doc_id, float("-inf")), _candidate_score(c))
        doc_candidates.setdefault(doc_id, []).append(c)

    ranked_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

    if len(ranked_docs) >= 2 and doc_scores[ranked_docs[0]] > 0:
        gap = (doc_scores[ranked_docs[0]] - doc_scores[ranked_docs[1]]) / abs(doc_scores[ranked_docs[0]])
        if gap >= confidence_gap:
            ranked_docs = ranked_docs[:1]  # high confidence -> single article only

    ranked_docs = ranked_docs[:max_documents]

    groups = []
    for doc_id in ranked_docs:
        chunks = sorted(doc_candidates[doc_id], key=lambda c: _candidate_score(c), reverse=True)
        chunks = chunks[:max_chunks_per_document]
        chunks = sorted(chunks, key=lambda c: c["chunk_text"])  # stable order for merging
        groups.append({
            "document_id": doc_id,
            "category": chunks[0]["category"],
            "source_file": chunks[0]["source_file"],
            "score": doc_scores[doc_id],
            "chunks": chunks,
        })
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_context_package(
    query,
    collection=None,
    bm25_bundle=None,
    reranker=None,
    top_k_initial=config.TOP_K_INITIAL,
    top_k_rerank=config.TOP_K_RERANK,
    max_documents=config.MAX_CONTEXT_DOCUMENTS,
    max_chunks_per_document=config.MAX_CHUNKS_PER_DOCUMENT,
    word_budget=config.WORD_BUDGET,
):
    """
    Same name/first-argument signature as before (`build_context_package(query)`
    still works), so streamlit_app.py doesn't break. Pass `collection`,
    `bm25_bundle`, and `reranker` from the app for cached, fast retrieval;
    if omitted they are built on the fly (slower, but self-contained).
    """
    if collection is None:
        collection = chroma_store.get_chroma_collection(reset=False)
    if bm25_bundle is None and config.USE_HYBRID:
        bm25_bundle = build_bm25_index(collection)

    candidates = hybrid_search(query, collection, bm25_bundle, k=top_k_initial)

    if config.USE_RERANKER:
        candidates = rerank_candidates(query, candidates, reranker, top_k=top_k_rerank)
    else:
        candidates = candidates[:top_k_rerank]

    if config.USE_MMR:
        candidates = apply_mmr(candidates)

    groups = group_by_document_with_voting(
        candidates,
        max_documents=max_documents,
        max_chunks_per_document=max_chunks_per_document,
    )

    blocks = []
    used_words = 0
    selected_flat = []
    for position, group in enumerate(groups, start=1):
        merged_text = " ".join(c["chunk_text"] for c in group["chunks"])
        words = len(merged_text.split())
        if blocks and used_words + words > word_budget:
            continue
        blocks.append(
            f"[Source {position}] category: {group['category']} | file: {group['source_file']}\n"
            f"{merged_text}"
        )
        used_words += words
        selected_flat.append({
            "document_id": group["document_id"],
            "category": group["category"],
            "source_file": group["source_file"],
            "chunk_text": merged_text,
            "score": group["score"],
        })

    return {
        "query": query,
        "candidates": candidates,          # post rerank/MMR, pre-grouping - for diagnostics
        "groups": groups,
        "selected": selected_flat,
        "context_text": "\n\n".join(blocks),
        "used_words": used_words,
        "num_sources": len(selected_flat),
    }


def main():
    sample_query = "What increased profits for the company this quarter?"
    package = build_context_package(sample_query)

    print(f"Query: {sample_query}")
    print(f"Sources selected: {package['num_sources']} | words used: {package['used_words']}")
    print()
    print(package["context_text"] or "(no context passed the score/word-budget filters)")
    print()
    print("Diagnostics:")
    for c in package["candidates"]:
        print(
            f"  chunk={c['chunk_id']:<18} doc={c['document_id']:<5} "
            f"dense={c['dense_score']} bm25={c['bm25_score']} "
            f"rrf={round(c['rrf_score'], 4)} rerank={c['rerank_score']}"
        )


if __name__ == "__main__":
    if not os.path.isdir(config.CHROMA_DIR) or not os.listdir(config.CHROMA_DIR):
        raise FileNotFoundError(
            f"'{config.CHROMA_DIR}' is empty. Run 05_create_chroma_store.py first."
        )
    main()
