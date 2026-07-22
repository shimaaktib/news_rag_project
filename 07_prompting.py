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
    """Grounded, citation-required prompt with strict anti-hallucination rules."""
    return f"""You are a precise question-answering assistant for BBC News articles.
You must answer ONLY using the numbered sources in the CONTEXT below.

STRICT RULES:
1. Use only facts stated in the CONTEXT. Never use outside knowledge or assumptions.
2. If the CONTEXT does not contain the answer, reply exactly:
   "The provided sources do not contain enough information to answer this question."
3. Do not invent names, numbers, dates, or quotes that are not in the CONTEXT.
4. Keep the answer concise (2-4 sentences) and directly focused on the question.
5. After every factual claim, cite the source number(s) it came from, like [Source 1].

OUTPUT FORMAT (exactly two sections):
Answer: <your grounded answer with inline [Source N] citations>
Sources: <comma-separated list of the source numbers you actually used>

QUESTION:
{query}

CONTEXT:
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
