"""Pre-tuning baseline: load Gemma 4 26B MoE and run the enrichment prompt on a few
diverse real core rows. Validates the stack (MoE on this GPU, multilingual
comprehension, JSON adherence) and shows how much tuning we actually need.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import SYSTEM, build_prompt, FIELDS

MODEL = "google/gemma-4-26B-A4B-it"


def parse_json(raw: str) -> dict:
    try:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])
    except Exception:
        return {}


def main() -> None:
    df = pd.read_parquet("enrichment/data/gold_seed_sample.parquet")
    # one row per (tradition, lang) combo for a spread, plus a Gita verse
    picks = []
    for (trad, lang), g in df.groupby(["tradition", "lang_original"]):
        picks.append(g.iloc[0])
    rows = pd.DataFrame(picks).head(8)

    print(f"loading {MODEL} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Blackwell (cc 10+): torch 2.8's grouped_mm MoE kernel only supports Hopper
    # (cc==9.0), so force the eager (looped) experts path. vLLM is used for bulk.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        experts_implementation="eager")
    print(f"loaded in {time.time()-t0:.0f}s | GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

    for _, row in rows.iterrows():
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + build_prompt(row.to_dict())}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        data = parse_json(raw)
        ok = all(data.get(k) for k in FIELDS[:3])
        print("\n" + "=" * 80)
        print(f"[{row.source} | {row.tradition} | {row.lang_original} | {row.text_type}] "
              f"{row.citation[:70]}")
        print(f"gen {time.time()-t0:.1f}s | valid_json={bool(data)} | core3_filled={ok}")
        for k in FIELDS:
            if k in data:
                print(f"  {k}: {data[k]}")
        if not data:
            print("  RAW:", raw[:300])


if __name__ == "__main__":
    main()
