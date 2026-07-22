"""
Step 1 — Documents
Loads the raw BBC News articles from data/bbc/<category>/*.txt into a single
table and saves it so the next steps can build on it.

Expected folder layout (download the BBC News dataset and place it here):
    data/bbc/business/*.txt
    data/bbc/entertainment/*.txt
    data/bbc/politics/*.txt
    data/bbc/sport/*.txt
    data/bbc/tech/*.txt
"""

import glob
import os

import pandas as pd

import config


def load_documents(data_dir=config.DATA_DIR):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Couldn't find '{data_dir}'. Download the BBC News dataset and place "
            f"its category folders (business/, entertainment/, politics/, sport/, tech/) "
            f"inside data/bbc/."
        )

    rows = []
    document_id = 0
    for category in sorted(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        for filepath in sorted(glob.glob(os.path.join(category_path, "*.txt"))):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            rows.append({
                "document_id": document_id,
                "category": category,
                "filename": os.path.basename(filepath),
                "text": text,
            })
            document_id += 1

    return pd.DataFrame(rows)


def main():
    documents_df = load_documents()
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    documents_df.to_csv(config.DOCUMENTS_CSV, index=False)

    print(f"Loaded {len(documents_df)} articles.")
    print(documents_df["category"].value_counts())
    print(f"Saved -> {config.DOCUMENTS_CSV}")


if __name__ == "__main__":
    main()
