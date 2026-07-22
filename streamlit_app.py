import streamlit as st

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


@st.cache_resource(show_spinner=False)
def ensure_index_ready():
    """
    Build the Chroma vector store the first time the app runs, straight from
    data/bbc/ — so nobody has to run 01_..05_.. by hand before the deployed
    app is usable. If the store was already built in a previous run (or is
    baked into the deployment), this just reuses it.
    """
    collection = chroma_store.get_chroma_collection(reset=False)
    if collection.count() > 0:
        return collection, collection.count()

    documents_df = documents_step.load_documents()
    preprocessed_df = preprocessing_step.preprocess_documents(documents_df)
    chunks_df = chunking_step.build_chunks(preprocessed_df)
    collection = chroma_store.build_store(chunks_df, reset=True)
    return collection, collection.count()


st.set_page_config(page_title="BBC News RAG Assistant", page_icon="📰")
st.title("📰 BBC News RAG Assistant")
st.caption("Ask a question about the BBC News articles indexed in the vector store.")

try:
    with st.spinner("Preparing the news index — first run only, this can take a minute..."):
        _, chunk_count = ensure_index_ready()
    st.caption(f"Index ready — {chunk_count} chunks indexed.")
except FileNotFoundError as e:
    st.error(
        f"{e}\n\nThe app can't build its index because the BBC News articles "
        f"aren't in this deployment. Add the `data/bbc/<category>/*.txt` files "
        f"to the repo and redeploy."
    )
    st.stop()

query = st.text_input("Your question", placeholder="e.g. What increased Time Warner's quarterly profits?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and query.strip():
    with st.spinner("Retrieving context..."):
        package = retrieve_context.build_context_package(query)

    if package["num_sources"] == 0:
        st.warning("No sources passed the relevance filters for this question. Try rephrasing it.")
    else:
        with st.spinner("Asking the model..."):
            answer = rag.ask_llm(query, package["context_text"])

        st.subheader("Answer")
        st.write(answer)

        with st.expander(f"Sources used ({package['num_sources']})"):
            st.text(package["context_text"])

elif ask_clicked:
    st.warning("Type a question first.")
