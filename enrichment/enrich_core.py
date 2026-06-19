"""A3 — bulk-enrich the counseling core with the QLoRA-tuned Gemma 4 26B MoE.

Loads the base model (bf16) + the trained LoRA adapter, batched generation over the
full 17,808-row core, parses the JSON, writes {id, 4 fields} incrementally to a jsonl
(resumable — re-running skips ids already done). Forces the eager MoE path (Blackwell).

    # quick throughput benchmark first (50 rows), check rows/sec + a sample:
    python -m enrichment.enrich_core --limit 50
    # full run:
    python -m enrichment.enrich_core --batch 16

Output: enrichment/data/enriched_core.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core_filter import select_core
from .prompt import SYSTEM, build_prompt, FIELDS

BASE = "google/gemma-4-26B-A4B-it"
ADAPTER = Path("enrichment/data/gemma4-enrich-lora")
CORPUS = Path("../satsangai/data/parquet/corpus.parquet")
OUT = Path("enrichment/data/enriched_core.jsonl")
COLS = ["id", "source", "tradition", "text_type", "citation", "ref", "lang_original",
        "original", "transliteration", "translation", "word_meanings", "commentaries"]

# Highest counseling value first — so limited GPU budget enriches these before the
# long narrative/biography tail. Anything not listed sorts after, in corpus order.
PRIORITY = ("vachanamrut", "swamini_vato", "satsang_diksha", "shikshapatri",
            "bhagavad_gita", "upanishads", "yoga_sutras", "satsangijivanam",
            "bhaktachintamani", "harililamrut", "aksharamrutam", "janmangal_namavali",
            "nishkulanand_kavya", "swaminarayan_bhashyam", "subbarao_gita_commentary",
            "chanakya_niti")


def parse_json(raw: str) -> dict:
    try:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--limit", type=int, default=None, help="benchmark on first N rows")
    ap.add_argument("--priority", action="store_true",
                    help="enrich highest-value counseling sources first")
    ap.add_argument("--retry-bad", action="store_true",
                    help="drop already-written rows that are missing core fields so they "
                         "get re-attempted (pair with a larger --max-new-tokens)")
    args = ap.parse_args()

    core = select_core(pd.read_parquet(CORPUS, columns=COLS))
    written = [json.loads(l) for l in OUT.read_text().splitlines()] if OUT.exists() else []
    if args.retry_bad:
        good = [r for r in written if all(r.get(k) for k in FIELDS[:3])]
        with open(OUT, "w") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"retry-bad: kept {len(good)} valid, dropped {len(written)-len(good)} for re-do")
        written = good
    done = {str(r["id"]) for r in written}
    core = core[~core["id"].astype(str).isin(done)]
    if args.priority:
        rank = {s: i for i, s in enumerate(PRIORITY)}
        core = core.assign(_p=core["source"].map(lambda s: rank.get(s, len(PRIORITY)))) \
                   .sort_values("_p", kind="stable").drop(columns="_p")
        print("priority order: enriching", ", ".join(PRIORITY[:5]), "... first")
    if args.limit:
        core = core.head(args.limit)
    print(f"to enrich: {len(core)} rows (already done: {len(done)})")
    if len(core) == 0:
        return

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"                      # left-pad for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda", experts_implementation="eager")
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model = model.merge_and_unload()    # fold LoRA into base weights -> faster inference
    model.eval()
    print(f"model+adapter merged | GPU {torch.cuda.memory_allocated()/1e9:.1f} GB")

    rows = core.to_dict("records")
    t0, n_ok, n_bad = time.time(), 0, 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i + args.batch]
            texts = [tok.apply_chat_template(
                        [{"role": "user", "content": SYSTEM + "\n\n" + build_prompt(r)}],
                        tokenize=False, add_generation_prompt=True) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            gen = out[:, enc["input_ids"].shape[1]:]
            for r, g in zip(batch, gen):
                data = parse_json(tok.decode(g, skip_special_tokens=True))
                ok = all(data.get(k) for k in FIELDS[:3])
                n_ok += ok
                n_bad += not ok
                f.write(json.dumps({"id": str(r["id"]),
                                    **{k: data.get(k) for k in FIELDS}},
                                   ensure_ascii=False) + "\n")
            f.flush()
            done_n = i + len(batch)
            rate = done_n / (time.time() - t0)
            print(f"  {done_n}/{len(rows)} | {rate:.2f} rows/s | "
                  f"ok {n_ok} bad {n_bad} | eta {(len(rows)-done_n)/rate/60:.0f} min")

    print(f"done: {n_ok} ok, {n_bad} unparsed -> {OUT}")


if __name__ == "__main__":
    main()
