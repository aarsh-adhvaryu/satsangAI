"""A4a — embed the enriched rows on the proposal's intended retrieval target
(`contextual_explanation` + `when_this_helps`) with BGE-M3 (same model + 1024-d
unit-norm as the KB). Writes enriched_core.parquet (id + 4 fields +
embedding_source_text + embedding) for review before the KB write-back.

Runs on GPU (~5 min) or CPU (~30-40 min) — auto-detects, override with --device.

    HF_HUB_OFFLINE=1 python -m enrichment.embed_core            # auto device
    HF_HUB_OFFLINE=1 python -m enrichment.embed_core --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .prompt import FIELDS

GOLD = Path("enrichment/data/enriched_core.jsonl")
OUT = Path("enrichment/data/enriched_core.parquet")
EMBED_MODEL = "BAAI/bge-m3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in GOLD.read_text().splitlines()]
    df = pd.DataFrame(rows)
    # only embed rows with real enrichment; the 4 null rows keep their KB embedding
    valid = df[df[list(FIELDS[:3])].notna().all(axis=1)].copy()
    valid["embedding_source_text"] = (valid["contextual_explanation"].fillna("") + "\n"
                                      + valid["when_this_helps"].fillna("")).str.strip()
    print(f"embedding {len(valid)} rows on {args.device} with {EMBED_MODEL}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=args.device)
    emb = model.encode(valid["embedding_source_text"].tolist(),
                       batch_size=args.batch, normalize_embeddings=True,
                       show_progress_bar=True)
    valid["embedding"] = emb.tolist()

    cols = ["id", *FIELDS, "embedding_source_text", "embedding"]
    valid[cols].to_parquet(OUT, index=False)
    print(f"wrote {len(valid)} embedded rows -> {OUT}  (dim {len(emb[0])})")


if __name__ == "__main__":
    main()
