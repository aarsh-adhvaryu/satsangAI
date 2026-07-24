"""Build the V1 retrieval index from the enriched KB.

Selects the counseling-core rows (via the app's core_filter) PLUS the shastrarth-breadth
acharya-school sources (Advaita/Vishishtadvaita/Dvaita/Shuddhadvaita commentaries), attaches
each row's embedding from embeddings.f32 (row-aligned), and writes one compact parquet the
API loads at startup. Counseling retrieval filters by tradition (swaminarayan + shared_hindu),
so the acharya schools are invisible to counseling and only surface in shastrarth mode
(retrieve() drops the tradition filter). The schools are unenriched, so their vector is the
BGE-M3 fallback on translation/original — fine for comparative retrieval.

    python -m api.build_index
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from enrichment.core_filter import load_manifest, select_core
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

    core_ids = set(select_core(corpus)["id"])
    # shastrarth breadth: the acharya-school commentary sources (kept out of counseling
    # by the tradition filter; only reachable in shastrarth mode). These are UNenriched,
    # UNtranslated raw Sanskrit OCR, so keep only rows with substantial Devanagari and no
    # mojibake — drops OCR junk (page numbers, English/TOC fragments). Grounding is then
    # genuine Sanskrit (the generator must read it; strong for Claude, weaker for Gemma).
    import re as _re
    _moji = _re.compile(r"[�]|Ã[\x80-\xbf]|â€|ï¿½")
    shastrarth_srcs = set(load_manifest().get("shastrarth_breadth", {}).get("sources") or [])
    is_school = corpus["source"].isin(shastrarth_srcs)

    def _substantial(t: str) -> bool:
        # keep real commentary (Sanskrit Devanagari OR English translation — both useful);
        # drop only OCR junk: tiny page-number/TOC fragments and mojibake.
        t = str(t)
        return len(t.strip()) >= 120 and not _moji.search(t)

    school_ok = is_school & corpus["original"].map(_substantial)
    keep_mask = corpus["id"].isin(core_ids) | school_ok
    pos = np.where(keep_mask.to_numpy())[0]
    idx = corpus.loc[pos, KEEP].reset_index(drop=True)
    idx["embedding"] = list(f32[pos].astype("float32"))   # enriched vector where available

    enriched = int(idx["contextual_explanation"].notna().sum())
    n_shastrarth = int(idx["source"].isin(shastrarth_srcs).sum())
    config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(config.INDEX_PATH, index=False)
    print(f"wrote {len(idx)} rows ({enriched} enriched, {n_shastrarth} shastrarth-breadth) "
          f"-> {config.INDEX_PATH}")
    print("traditions:", idx.tradition.value_counts().to_dict())


if __name__ == "__main__":
    main()
