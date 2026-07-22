"""
Step 2 — Preprocessing
Light text cleaning applied before chunking: lowercasing, collapsing
whitespace, and stripping characters that aren't useful for search or
embeddings.
"""

import os
import re

import pandas as pd

import config


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-.,?!']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_documents(documents_df):
    documents_df = documents_df.copy()
    documents_df["clean_text"] = documents_df["text"].map(normalize_text)
    return documents_df


def main():
    if not os.path.exists(config.DOCUMENTS_CSV):
        raise FileNotFoundError(
            f"'{config.DOCUMENTS_CSV}' not found. Run 01_documents.py first."
        )

    documents_df = pd.read_csv(config.DOCUMENTS_CSV)
    preprocessed_df = preprocess_documents(documents_df)

    preprocessed_df.to_csv(config.PREPROCESSED_CSV, index=False)
    print(f"Preprocessed {len(preprocessed_df)} documents.")
    print(preprocessed_df[["text", "clean_text"]].head(2))
    print(f"Saved -> {config.PREPROCESSED_CSV}")


if __name__ == "__main__":
    main()
