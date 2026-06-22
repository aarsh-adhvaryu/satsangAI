"""Build the V1 counseling-core retrieval index from the enriched KB.

Selects the counseling-core rows (via the app's core_filter), attaches each row's
embedding from embeddings.f32 (row-aligned), and writes a single compact parquet the
API loads at startup: text + enrichment + metadata + 1024-d embedding.

    python -m api.build_index
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from enrichment.core_filter import select_core
from . import config

KEEP = ["id", "source", "text_type", "tradition", "citation", "ref", "lang_original",
        "original", "transliteration", "translation", "word_meanings", "commentaries",
        "contextual_explanation", "when_this_helps", "core_principle",
        "gujarati_explanation", "embedding_source_text", "verified"]


def main() -> None:
    meta = json.loads(config.KB_META.read_text())
    n, d = meta["rows"], meta["dim"]
    corpus = pd.read_parquet(config.KB_CORPUS).reset_index(drop=True)  # row order == f32
    f32 = np.memmap(config.KB_F32, dtype="float32", mode="r", shape=(n, d))

    core_mask = corpus["id"].isin(select_core(corpus)["id"])
    pos = np.where(core_mask.to_numpy())[0]
    idx = corpus.loc[pos, KEEP].reset_index(drop=True)
    idx["embedding"] = list(f32[pos].astype("float32"))   # enriched vector where available

    enriched = idx["contextual_explanation"].notna().sum()
    config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(config.INDEX_PATH, index=False)
    print(f"wrote {len(idx)} core rows ({enriched} enriched) -> {config.INDEX_PATH}")
    print("traditions:", idx.tradition.value_counts().to_dict())


if __name__ == "__main__":
    main()
