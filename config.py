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
# Context building
# ---------------------------------------------------------------------------
RETRIEVAL_K = 8              # how many candidates to pull from the vector store
MAX_CONTEXT_CHUNKS = 3        # how many chunks actually go into the final prompt
MAX_CHUNKS_PER_DOCUMENT = 1   # keep the context diverse across articles
WORD_BUDGET = 150             # cap on total words inside the context

# ---------------------------------------------------------------------------
# OpenRouter (LLM) — never hardcode a real key here.
# Locally it is read from a .env file; on Streamlit Cloud it is read from
# st.secrets inside streamlit_app.py.
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
