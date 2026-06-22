"""Retrieval-lift eval — does enrichment-based retrieval beat the old translation-based
retrieval? Isolates the EMBEDDING change (no rerank): same BGE-M3 query vector, searched
against the OLD vectors (embeddings.f32.pre_enrich.bak) vs the NEW enrichment vectors.

For each counseling query we take top-k from each, show both sets blind (A/B randomized)
to an Opus judge, and ask which set better helps the person. Reports a win rate.

    source ~/.zshrc && HF_HUB_OFFLINE=1 python -m eval.retrieval_lift
"""
from __future__ import annotations

import functools
import json
import random

import numpy as np
import pandas as pd

from api import config
from api.embed import embed_query
from enrichment.core_filter import select_core

QUERIES = [
    "I keep losing my temper with my family and feel guilty afterward",
    "I'm crushed by grief after losing my spouse",
    "I feel intense jealousy when others succeed",
    "I prayed for years and feel God never answered; I'm losing faith",
    "the pressure at work is overwhelming me and I feel like a failure",
    "I can't forgive someone who betrayed me and it's consuming me",
    "I feel completely alone and disconnected from everyone",
    "I'm anxious about the future and can't stop worrying",
    "I feel like my life has no purpose or meaning",
    "I'm struggling to control my desires and cravings",
]
TOP_K = 5
JUDGE_MODEL = "claude-opus-4-8"


@functools.lru_cache(maxsize=1)
def _setup():
    meta = json.loads((config.KB / "embeddings_meta.json").read_text())
    n, d = meta["rows"], meta["dim"]
    corpus = pd.read_parquet(config.KB_CORPUS).reset_index(drop=True)
    pos = np.where(corpus["id"].isin(select_core(corpus)["id"]).to_numpy())[0]
    meta_df = corpus.loc[pos, ["id", "citation", "source", "tradition", "translation",
                               "original", "contextual_explanation"]].reset_index(drop=True)
    after = np.memmap(config.KB_F32, dtype="float32", mode="r", shape=(n, d))[pos].astype("float32")
    before = np.memmap(config.KB / "embeddings.f32.pre_enrich.bak", dtype="float32",
                       mode="r", shape=(n, d))[pos].astype("float32")
    trad = meta_df["tradition"].isin(config.COUNSELING_TRADITIONS).to_numpy()
    return meta_df, before, after, trad


def _topk(qv, emb, trad):
    s = emb @ qv
    s = np.where(trad, s, -1.0)
    idx = np.argsort(-s)[:TOP_K]
    return idx


def _block(meta_df, idxs) -> str:
    out = []
    for i in idxs:
        r = meta_df.iloc[int(i)]
        body = (str(r.translation or r.original) or "")[:200]
        out.append(f"- {r.citation}: {body} | meaning: {str(r.contextual_explanation or '')[:160]}")
    return "\n".join(out)


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


SCHEMA = {"type": "object", "properties": {
    "winner": {"type": "string", "enum": ["A", "B", "tie"]}, "reason": {"type": "string"}},
    "required": ["winner", "reason"], "additionalProperties": False}


def _judge(query, block_a, block_b):
    content = (f"A person needs help with: \"{query}\"\n\nTwo retrieval systems each "
               f"returned {TOP_K} scripture passages. Which set is MORE relevant and "
               f"helpful for this person's actual need?\n\nSET A:\n{block_a}\n\nSET B:\n{block_b}")
    r = _client().messages.create(model=JUDGE_MODEL, max_tokens=400,
        system="You are a strict evaluator of retrieval relevance for a counseling "
               "companion. Judge which set better matches the person's real need. "
               "Return strict JSON.",
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    return json.loads(next(b.text for b in r.content if b.type == "text"))


def main() -> None:
    meta_df, before, after, trad = _setup()
    rng = random.Random(0)
    wins = {"after": 0, "before": 0, "tie": 0}
    print(f"{'query':52} winner")
    print("-" * 70)
    for q in QUERIES:
        qv = embed_query(q).astype("float32")
        a_idx, b_idx = _topk(qv, after, trad), _topk(qv, before, trad)
        # blind randomize which slot (A/B) holds the enriched set
        after_is_A = rng.random() < 0.5
        block_A = _block(meta_df, a_idx if after_is_A else b_idx)
        block_B = _block(meta_df, b_idx if after_is_A else a_idx)
        res = _judge(q, block_A, block_B)
        w = res["winner"]
        winner = "tie" if w == "tie" else ("after" if (w == "A") == after_is_A else "before")
        wins[winner] += 1
        print(f"{q[:52]:52} {winner}")
    n = len(QUERIES)
    print(f"\n=== RETRIEVAL LIFT (enriched vs translation-based, blind A/B) ===")
    print(f"  enriched WINS : {wins['after']}/{n}")
    print(f"  old WINS      : {wins['before']}/{n}")
    print(f"  ties          : {wins['tie']}/{n}")


if __name__ == "__main__":
    main()
