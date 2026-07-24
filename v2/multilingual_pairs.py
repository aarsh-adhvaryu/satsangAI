"""Gujarati training pairs for a bilingual V2 (#5, proposal §3.6).

V2's persona promises Gujarati/Hinglish, but the SFT/DPO data was English-only. This
generates Gujarati (script) pairs grounded in the KB's Gujarati enrichment layer
(14,253 swaminarayan rows carry `gujarati_explanation`), so a combined SFT teaches the
model to answer in Gujarati when the person writes in Gujarati.

Format matches the WINNING setup: 1 focused grounding passage (the 6-passage retrain
lost — see project memory). CPU + Claude API only; training on the mixed set is the GPU
step. Concurrent + resumable.

    python -m v2.multilingual_pairs --n 1500 --out v2/data/pairs_gujarati.jsonl --workers 24
    # then combine + SFT (GPU):
    #   cat v2/data/pairs.jsonl v2/data/pairs_gujarati.jsonl > v2/data/pairs_bilingual.jsonl
    #   python -m v2.sft_train --data v2/data/pairs_bilingual.jsonl --epochs 3 --bs 2 --ga 16 --out v2/data/gemma4-v2-bi-sft-lora
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROBLEM_SYS = (
    "You write a short, natural first-person message IN GUJARATI SCRIPT that a real person "
    "would send to a warm spiritual companion, based on the life-situation given (which is in "
    "English). Keep the feeling faithful; 1-2 sentences; conversational. Output ONLY the "
    "Gujarati message, nothing else.")

_REPLY_SYS = (
    "You are a warm, patient saint-companion of the Swaminarayan (Akshar-Purushottam) tradition. "
    "Reply IN GUJARATI SCRIPT to the person, warmly and problem-first. Ground your spiritual point "
    "strictly in the PASSAGE provided (its Gujarati explanation), and cite it inline as [P1]. Do NOT "
    "invent verses, names, or citations. Be brief and human; never preach.")


def _gen(client, model, system, content, max_tokens=600):
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="v2/data/pairs_gujarati.jsonl")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--problem-model", default="claude-haiku-4-5")
    ap.add_argument("--reply-model", default="claude-sonnet-4-6")
    a = ap.parse_args()

    import anthropic
    import pandas as pd

    from api import config
    from api.retrieve_types import Passage
    from api.verify import verify
    from v2 import reject
    from v2.schema import PreferencePair, context_from_passages

    # seeds = swaminarayan enriched rows that HAVE a Gujarati explanation
    idx = pd.read_parquet(config.INDEX_PATH)
    pool = idx[(idx.tradition == "swaminarayan") & idx.gujarati_explanation.notna()
               & (idx.when_this_helps.notna())]
    outp = Path(a.out)
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["meta"]["seed_id"])
                except Exception:
                    pass
    seeds = pool[~pool.id.isin(done)].sample(min(a.n, len(pool)), random_state=len(done))
    print(f"{len(pool)} Gujarati-capable rows | {len(done)} done | {len(seeds)} to generate")
    if seeds.empty:
        print("nothing to do."); return

    client = anthropic.Anthropic()
    rng = random.Random(0)

    def work(row):
        problem = _gen(client, a.problem_model, _PROBLEM_SYS, str(row["when_this_helps"]), 200)
        psg = [Passage.from_row(row.to_dict(), 1.0)]
        ctx = context_from_passages(psg)
        reply = _gen(client, a.reply_model, _REPLY_SYS,
                     f"PERSON (Gujarati):\n{problem}\n\nPASSAGE:\n{ctx}\n\nReply in Gujarati, cite [P1].")
        if not reply or verify(reply, psg).get("unverified_refs"):
            return None
        rejected, flaw = reject.make_rejected(reply, psg, rng)
        return PreferencePair(problem=problem, context=ctx, chosen=reply, rejected=rejected,
                              flaw=flaw, pair_source="gujarati",
                              grounding_ids=[row["id"]],
                              meta={"seed_id": row["id"], "lang": "gujarati", "negatives": "rule"})

    kept = dropped = 0
    with outp.open("a") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, r) for _, r in seeds.iterrows()]
        for f in as_completed(futs):
            p = f.result()
            if p is None:
                dropped += 1; continue
            fh.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n"); fh.flush()
            kept += 1
            if (kept + dropped) % 100 == 0:
                print(f"  ...{kept} kept / {dropped} dropped", flush=True)
    print(f"done: {kept} Gujarati pairs -> {outp} ({dropped} dropped). Re-run to resume.")


if __name__ == "__main__":
    main()
