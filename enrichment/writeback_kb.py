"""A4b — write enrichment + new embeddings back into the KB source-of-truth files.

For the 17,804 enriched core rows, fills the 4 enrichment text fields +
embedding_source_text in ../satsangai/data/parquet/corpus.parquet, and overwrites
those same rows (by corpus row order) in embeddings.f32 with the new
enrichment-based BGE-M3 vectors. The other ~214k rows are untouched.

Backs up corpus.parquet + embeddings.f32 (once) before mutating. Does NOT push to
HF — review the result, then run push_hf separately.

    python -m enrichment.writeback_kb            # dry-run summary only
    python -m enrichment.writeback_kb --apply    # back up + mutate the KB files
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .prompt import FIELDS

ENRICHED = Path("enrichment/data/enriched_core.parquet")
KB = Path("../satsangai/data/parquet")
CORPUS = KB / "corpus.parquet"
F32 = KB / "embeddings.f32"
META = KB / "embeddings_meta.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually mutate the KB files")
    args = ap.parse_args()

    enr = pd.read_parquet(ENRICHED)
    enr["id"] = enr["id"].astype(str)
    emb = np.asarray(enr["embedding"].tolist(), dtype="float32")  # (N,1024) unit-norm
    meta = json.loads(META.read_text())
    dim = meta["dim"]
    assert emb.shape[1] == dim, f"dim mismatch {emb.shape[1]} != {dim}"

    corpus = pd.read_parquet(CORPUS).reset_index(drop=True)   # row order == f32 order
    corpus["id"] = corpus["id"].astype(str)
    pos = {cid: i for i, cid in enumerate(corpus["id"])}
    rows = enr[enr["id"].isin(pos)].copy()
    missing = len(enr) - len(rows)
    print(f"enriched rows: {len(enr)} | matched in corpus: {len(rows)} | unmatched: {missing}")
    print(f"corpus rows: {len(corpus)} | f32: {meta['rows']}x{dim}")
    print(f"fields to write: {list(FIELDS)} + embedding_source_text + embedding(.f32)")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to back up + mutate. No HF push here.")
        return

    # 1. one-time backup of the KB source-of-truth files
    for f in (CORPUS, F32):
        bak = f.with_suffix(f.suffix + ".pre_enrich.bak")
        if not bak.exists():
            print(f"backing up {f.name} -> {bak.name}")
            shutil.copy2(f, bak)

    # 2. text fields into corpus.parquet (only for matched ids)
    em = rows.set_index("id")
    idx = corpus.index[corpus["id"].isin(em.index)]
    for col in [*FIELDS, "embedding_source_text"]:
        corpus.loc[idx, col] = corpus.loc[idx, "id"].map(em[col])
    corpus.to_parquet(CORPUS, index=False)
    print(f"updated {len(idx)} rows in {CORPUS.name}")

    # 3. new vectors into embeddings.f32 at the matching row positions
    mm = np.memmap(F32, dtype="float32", mode="r+", shape=(meta["rows"], dim))
    positions = rows["id"].map(pos).to_numpy()
    mm[positions] = emb
    mm.flush()
    print(f"overwrote {len(positions)} vectors in {F32.name}")
    print("\nDONE. Review, then push: "
          "cd ../satsangai && python -m pipeline.push_hf "
          "--repo aarsh-adhvaryu/satsangai-kb --private")


if __name__ == "__main__":
    main()
