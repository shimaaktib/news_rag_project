"""
Step 7 — Prompting
Builds a grounded, citation-required prompt from a question + retrieved
context, then sends it to an LLM through OpenRouter.

API key handling:
- Locally: put OPENROUTER_API_KEY (and optionally OPENROUTER_MODEL) in a
  .env file. NEVER commit the real .env file or hardcode the key here.
- On Streamlit Cloud: streamlit_app.py fills these in from st.secrets at
  runtime (see the try/except block there) — that's why this module keeps
  them as plain module-level variables it's fine to overwrite from outside.
"""

import os

import requests
from dotenv import load_dotenv

import config

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", config.DEFAULT_OPENROUTER_MODEL)


def build_prompt(query, context_text):
    """
    Strict, grounded prompt style — forces a checkable, cited answer and
    stops the model from silently blending facts from different articles.
    Each [Source N] block in `context_text` is now ONE article (its chunks
    are already merged by 06_retrieve_context.py's document grouping), so
    "prefer one source" maps directly onto "prefer one article".
    """
    return f"""You are a grounded news question-answering assistant.

Rules:
1. Use only the provided context (CONTEXT). Never add outside knowledge.
2. If the answer is not in the context, say exactly: "The provided sources do not contain enough information to answer this question."
3. Each [Source N] block is a separate news article. Prefer answering from a single source whenever it fully answers the question.
4. Do NOT merge facts from different sources into one narrative unless the question explicitly asks for a comparison across sources.
5. If the sources disagree or describe different dates/events, say so explicitly instead of blending them into a single answer — e.g. "Source 1 and Source 2 report different developments; ..."
6. Cite every claim inline with its source number, e.g. [Source 1].

OUTPUT FORMAT (exactly two sections):
Answer: [your grounded answer, with inline [Source N] citations]
Sources: [comma-separated source numbers actually used, e.g. Source 1, Source 2]

Question:
{query}

Context:
{context_text}
"""


def ask_llm(query, context_text):
    if not OPENROUTER_API_KEY:
        return (
            "No OPENROUTER_API_KEY configured. Add it to a local .env file, "
            "or to your Streamlit Cloud app's Secrets."
        )

    prompt = build_prompt(query, context_text)

    response = requests.post(
        config.OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def main():
    sample_context = (
        "[Source 1] category: business | file: 001.txt\n"
        "Time Warner's quarterly profits rose thanks to higher advertising sales "
        "and a boost from its stake in Google."
    )
    sample_query = "What increased Time Warner's quarterly profits?"

    print(build_prompt(sample_query, sample_context))
    print("-" * 60)
    print(ask_llm(sample_query, sample_context))


if __name__ == "__main__":
    main()
