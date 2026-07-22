"""
Shared configuration for the RAG pipeline.
Every numbered script (01_..07_..) and streamlit_app.py reads its settings
from here so the whole project stays consistent end-to-end.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "bbc")            # raw articles: data/bbc/<category>/*.txt
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # intermediate CSVs between steps

DOCUMENTS_CSV = os.path.join(PROCESSED_DIR, "01_documents.csv")
PREPROCESSED_CSV = os.path.join(PROCESSED_DIR, "02_preprocessed.csv")
CHUNKS_CSV = os.path.join(PROCESSED_DIR, "03_chunks.csv")
EMBEDDINGS_NPY = os.path.join(PROCESSED_DIR, "04_embeddings.npy")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")
CHROMA_COLLECTION_NAME = "bbc_news_chunks"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 60      # words per chunk
CHUNK_OVERLAP = 15   # overlapping words between consecutive chunks

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval quality controls
# ---------------------------------------------------------------------------
USE_HYBRID = True             # fuse dense (Chroma) + lexical (BM25) search
USE_RERANKER = True           # cross-encoder re-ranking of fused candidates
USE_MMR = True                # diversify the re-ranked candidates (MMR)

TOP_K_INITIAL = 20            # candidates pulled per method before fusion
TOP_K_RERANK = 8              # candidates kept after cross-encoder re-ranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60                    # Reciprocal Rank Fusion constant
MMR_LAMBDA = 0.7              # 1.0 = pure relevance, 0.0 = pure diversity

SIMILARITY_THRESHOLD = 0.15   # min dense cosine score to enter the dense candidate list
CONFIDENCE_GAP = 0.15         # if top doc beats 2nd doc by this fraction, use ONE document only

# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------
RETRIEVAL_K = TOP_K_INITIAL    # kept for backward compatibility with older calls
MAX_CONTEXT_DOCUMENTS = 3      # how many distinct articles may appear in the context
MAX_CHUNKS_PER_DOCUMENT = 3    # chunks merged together per article ("context grouping")
WORD_BUDGET = 220              # cap on total words inside the context
MAX_CONTEXT_CHUNKS = MAX_CONTEXT_DOCUMENTS  # legacy alias, some old code still reads this

# ---------------------------------------------------------------------------
# OpenRouter (LLM) — never hardcode a real key here.
# Locally it is read from a .env file; on Streamlit Cloud it is read from
# st.secrets inside streamlit_app.py.
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
