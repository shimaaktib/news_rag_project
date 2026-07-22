import datetime

import streamlit as st

import config
from _pipeline_utils import load_module

documents_step = load_module("01_documents.py", "documents_step")
preprocessing_step = load_module("02_preprocessing.py", "preprocessing_step")
chunking_step = load_module("03_chunking.py", "chunking_step")
chroma_store = load_module("05_create_chroma_store.py", "chroma_store")
retrieve_context = load_module("06_retrieve_context.py", "retrieve_context")
rag = load_module("07_prompting.py", "rag")

# On Streamlit Cloud there's no .env file, so pull the key/model from the
# app's Secrets instead. Locally, rag already picked these up from .env.
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Index bootstrap (runs once, cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_index_ready():
    collection = chroma_store.get_chroma_collection(reset=False)
    if collection.count() > 0:
        return collection, collection.count()

    documents_df = documents_step.load_documents()
    preprocessed_df = preprocessing_step.preprocess_documents(documents_df)
    chunks_df = chunking_step.build_chunks(preprocessed_df)
    collection = chroma_store.build_store(chunks_df, reset=True)
    return collection, collection.count()


@st.cache_resource(show_spinner=False)
def ensure_bm25_ready():
    """Lexical (BM25) index used for hybrid retrieval. Built once from Chroma."""
    if not config.USE_HYBRID:
        return None
    collection = chroma_store.get_chroma_collection(reset=False)
    return retrieve_context.build_bm25_index(collection)


@st.cache_resource(show_spinner=False)
def ensure_reranker_ready():
    """Cross-encoder re-ranking model, loaded once and cached across reruns."""
    if not config.USE_RERANKER:
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BBC News RAG Assistant",
    page_icon="News",
    layout="centered",
)

CATEGORY_COLORS = {
    "business": "#1f6f3d",
    "entertainment": "#8e44ad",
    "politics": "#b0341c",
    "sport": "#1a6091",
    "tech": "#0b5563",
}

ACCENT = "#bb1919"  # BBC red

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

    .stApp {{ background: #ffffff; }}
    .block-container {{ padding-top: 2.2rem; max-width: 780px; }}

    html, body, .stMarkdown, p, span, div, li {{
        font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    }}

    .hero {{ text-align: center; padding: 6px 0 2px 0; }}
    .hero-brand {{ display: inline-flex; align-items: center; gap: 9px; margin-bottom: 12px; }}
    .hero-logo {{
        background: {ACCENT}; color: #fff; font-family: 'Inter', sans-serif;
        font-weight: 700; font-size: 15px; letter-spacing: 2px; padding: 5px 10px; border-radius: 3px;
    }}
    .hero-brand-text {{
        font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13px;
        letter-spacing: 3px; text-transform: uppercase; color: #555;
    }}
    .hero-title {{
        font-family: 'Playfair Display', Georgia, serif; font-weight: 900; font-size: 52px;
        line-height: 1.02; color: #0a0a0a; margin: 0; letter-spacing: -0.5px;
    }}
    .hero-sub {{
        font-family: 'Source Serif 4', serif; font-style: italic; font-size: 18px;
        color: #555; margin: 12px 0 0 0;
    }}
    .hero-divider {{
        height: 3px; margin: 20px 0 4px 0;
        background: linear-gradient(90deg, transparent, {ACCENT} 20%, {ACCENT} 80%, transparent);
    }}

    .stat-strip {{ display: flex; justify-content: center; gap: 0; margin: 18px 0 26px 0; }}
    .stat {{ text-align: center; padding: 0 26px; border-right: 1px solid #e6e6e6; }}
    .stat:last-child {{ border-right: none; }}
    .stat-num {{
        font-family: 'Playfair Display', serif; font-weight: 800; font-size: 26px;
        color: {ACCENT}; line-height: 1;
    }}
    .stat-label {{
        font-family: 'Inter', sans-serif; font-size: 11px; letter-spacing: 1.5px;
        text-transform: uppercase; color: #888; margin-top: 5px;
    }}

    .section-label {{
        font-family: 'Inter', sans-serif; font-weight: 600; font-size: 12px;
        letter-spacing: 2.5px; text-transform: uppercase; color: {ACCENT};
        display: flex; align-items: center; gap: 10px; margin: 8px 0 12px 0;
    }}
    .section-label::after {{ content: ""; flex: 1; height: 1px; background: #e6e6e6; }}

    .answer-card {{
        background: #fcfbf9; border: 1px solid #ece7dd; border-left: 5px solid {ACCENT};
        border-radius: 4px; padding: 24px 28px; font-size: 20px; line-height: 1.75;
        color: #1a1a1a; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .answer-card p {{ margin: 0; }}

    .source-card {{
        border: 1px solid #ece7dd; background: #ffffff; border-radius: 4px;
        padding: 14px 18px; margin-bottom: 12px; transition: box-shadow .15s ease;
    }}
    .source-card:hover {{ box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    .source-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
    .cat-badge {{
        color: #fff; font-family: 'Inter', sans-serif; font-size: 10px; font-weight: 600;
        letter-spacing: 1.2px; text-transform: uppercase; padding: 3px 10px; border-radius: 3px;
    }}
    .source-file {{ font-family: 'Inter', sans-serif; color: #999; font-size: 12px; }}
    .source-text {{ font-size: 15px; color: #333; line-height: 1.6; }}

    div.stButton > button {{
        background: {ACCENT}; color: #fff; border: none; border-radius: 3px;
        font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: 0.5px;
        padding: 8px 40px; transition: background .15s ease;
    }}
    div.stButton > button:hover {{ background: #990f0f; color: #fff; }}

    .stTextInput input {{
        font-family: 'Source Serif 4', Georgia, serif; font-size: 18px;
        padding: 12px 14px; border-radius: 4px;
    }}

    .app-footer {{
        text-align: center; margin-top: 40px; padding-top: 18px; border-top: 1px solid #eee;
        font-family: 'Inter', sans-serif; font-size: 12px; color: #aaa; letter-spacing: 0.5px;
    }}
    .app-footer b {{ color: #777; }}

    header[data-testid="stHeader"] {{ background: transparent; }}
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-brand">
            <span class="hero-logo">BBC</span>
            <span class="hero-brand-text">News Intelligence</span>
        </div>
        <div class="hero-title">The Newsroom<br>Answers</div>
        <div class="hero-sub">Ask anything about the BBC News archive &mdash; answers grounded in real reporting, with every source cited.</div>
        <div class="hero-divider"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Index status + stat strip
# ---------------------------------------------------------------------------
try:
    with st.spinner("Indexing the newsroom archive - first run only, please wait..."):
        collection, chunk_count = ensure_index_ready()
        bm25_bundle = ensure_bm25_ready()
        reranker = ensure_reranker_ready()
except FileNotFoundError as e:
    st.error(
        f"{e}\n\nThe app can't build its index because the BBC News articles "
        f"aren't in this deployment. Add the data/bbc/<category>/*.txt files "
        f"to the repo and redeploy."
    )
    st.stop()

st.markdown(
    f"""
    <div class="stat-strip">
        <div class="stat">
            <div class="stat-num">{chunk_count:,}</div>
            <div class="stat-label">Passages Indexed</div>
        </div>
        <div class="stat">
            <div class="stat-num">5</div>
            <div class="stat-label">News Sections</div>
        </div>
        <div class="stat">
            <div class="stat-num">100%</div>
            <div class="stat-label">Source-Cited</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Query box + clickable examples
# ---------------------------------------------------------------------------
if "query_value" not in st.session_state:
    st.session_state.query_value = ""

st.markdown('<div class="section-label">Ask the Newsroom</div>', unsafe_allow_html=True)

examples = [
    "What increased Time Warner's profits?",
    "Why did IBM invest more in Linux?",
    "What was said about the UK economy?",
]
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example, key=f"ex_{example}", use_container_width=True):
        st.session_state.query_value = example

query = st.text_input(
    "Your question",
    value=st.session_state.query_value,
    placeholder="Type your question about the BBC News archive...",
    label_visibility="collapsed",
    key="query_input",
)
ask_clicked = st.button("Ask the archive", type="primary")

# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------
if ask_clicked and query.strip():
    with st.spinner("Searching the archive for relevant reporting..."):
        package = retrieve_context.build_context_package(
            query,
            collection=collection,
            bm25_bundle=bm25_bundle,
            reranker=reranker,
        )

    if package["num_sources"] == 0:
        st.warning("No passages passed the relevance filters for this question. Try rephrasing it.")
    else:
        with st.spinner("Composing a grounded answer..."):
            answer = rag.ask_llm(query, package["context_text"])

        st.markdown('<div class="section-label">The Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card"><p>{answer}</p></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-label" style="margin-top:26px;">Sources Cited</div>',
            unsafe_allow_html=True,
        )
        for i, row in enumerate(package["selected"], start=1):
            color = CATEGORY_COLORS.get(str(row["category"]).lower(), "#444")
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-head">
                        <span class="cat-badge" style="background:{color};">{row['category']}</span>
                        <span class="source-file">Source {i} &middot; {row['source_file']}</span>
                    </div>
                    <div class="source-text">{row['chunk_text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Retrieval diagnostics (dense / BM25 / RRF / rerank scores)"):
            diagnostics_rows = [
                {
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "source_file": c["source_file"],
                    "dense_score": round(c["dense_score"], 4) if c["dense_score"] is not None else None,
                    "bm25_score": round(c["bm25_score"], 4) if c["bm25_score"] is not None else None,
                    "rrf_score": round(c["rrf_score"], 4) if c["rrf_score"] is not None else None,
                    "rerank_score": round(c["rerank_score"], 4) if c["rerank_score"] is not None else None,
                }
                for c in package["candidates"]
            ]
            st.dataframe(diagnostics_rows, use_container_width=True, hide_index=True)

elif ask_clicked:
    st.warning("Type a question first.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
year = datetime.date.today().year
st.markdown(
    f"""
    <div class="app-footer">
        Built with <b>Retrieval-Augmented Generation</b> &nbsp;&middot;&nbsp;
        Embeddings + Chroma vector store &nbsp;&middot;&nbsp;
        Answers grounded in retrieved context &nbsp;&middot;&nbsp; {year}
    </div>
    """,
    unsafe_allow_html=True,
)
