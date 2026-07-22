# BBC News RAG Assistant

A simple retrieval-augmented generation project: raw news articles → preprocessing →
chunking → vector embeddings → Chroma vector store → context retrieval → grounded
prompting → Streamlit assistant.

## Pipeline

| File | Step |
|---|---|
| `01_documents.py` | Load raw `.txt` articles from `data/bbc/<category>/` |
| `02_preprocessing.py` | Clean and normalize the text |
| `03_chunking.py` | Split articles into overlapping chunks |
| `04_vector_representation.py` | Embed chunks with `all-MiniLM-L6-v2` |
| `05_create_chroma_store.py` | Build the persistent Chroma vector store |
| `06_retrieve_context.py` | Query the store, assemble a filtered context package |
| `07_prompting.py` | Build the grounded prompt and call the LLM via OpenRouter |
| `streamlit_app.py` | The user-facing chat UI |

## 1. Get the data

Download the BBC News dataset and place it here so the folders line up:

```
data/bbc/business/*.txt
data/bbc/entertainment/*.txt
data/bbc/politics/*.txt
data/bbc/sport/*.txt
data/bbc/tech/*.txt
```

## 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Set your API key locally

Copy `.env.example` to `.env` and fill in your real key. **Never commit `.env`.**

```bash
cp .env.example .env
```

## 4. Run the app

**No manual pipeline run is required.** `streamlit_app.py` builds the index itself,
the first time it's asked a question — it calls steps 01→05
(`load_documents → preprocess_documents → build_chunks → build_store`) in memory and
caches the result with `st.cache_resource`. If a `chroma_store/` already exists with
data in it, that build is skipped and the existing store is reused instead.

```bash
streamlit run streamlit_app.py
```

The numbered scripts (`01_documents.py` … `07_prompting.py`) still work standalone too
(`python 03_chunking.py`, etc.) if you want to inspect or debug one step at a time —
they're just no longer a required manual step before the app works.

## 5. Deploy to Streamlit Cloud

1. Make sure `data/bbc/<category>/*.txt` (the actual BBC News articles) are committed
   to the repo — the app needs the raw data to build its own index on first run.
   Everything else it generates (`data/processed/`, `chroma_store/`) stays git-ignored
   and gets rebuilt automatically.
2. Push this project to a GitHub repo (`.env` stays out of it — check `.gitignore`).
3. On [share.streamlit.io](https://share.streamlit.io), create an app from that repo,
   entry point `streamlit_app.py`.
4. In the deployed app, open **Manage app → Secrets** and add:

   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```

5. Open the app. The first load will show "Preparing the news index..." while it builds
   the vector store — that's expected and only happens once per deployment.

## Final submission checklist

- [ ] All required Python files exist (`01_..07_..`, `streamlit_app.py`, `requirements.txt`)
- [ ] Real API key is **not** in the ZIP or the GitHub repo
- [ ] Streamlit Cloud secrets are set in valid TOML
- [ ] The deployed app runs and answers a question
- [ ] The answer uses retrieved context
- [ ] The answer cites sources
