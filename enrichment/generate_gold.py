"""Generate the QLoRA gold set OFFLINE with Claude (distillation seed only).

Per the project rule: Claude never enriches the shared KB at runtime; it is used
ONCE, offline, to produce a bounded gold set (~1.5k diverse rows) that QLoRA-tunes
the local Gemma 4 26B MoE. Runtime enrichment stays Gemma-only. See
config/counseling_core.yaml and the enrichment memory.

Uses the Batch API (50% cheaper, async) with the shared system prompt cached.
Strict JSON is enforced with structured outputs so every gold row is parseable.

Two steps:
    # 1. submit — needs ANTHROPIC_API_KEY
    python -m enrichment.generate_gold submit
    # 2. collect — poll + write gold.jsonl
    python -m enrichment.generate_gold collect --batch-id msgbatch_xxx
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .prompt import SYSTEM, build_prompt, FIELDS

MODEL = "claude-opus-4-8"          # best gold quality; offline/tuning only
SAMPLE = Path("enrichment/data/gold_seed_sample.parquet")
GOLD = Path("enrichment/data/gold.jsonl")
BATCH_META = Path("enrichment/data/gold_batch.json")

# Structured-output schema: 3 required fields + optional Gujarati.
SCHEMA = {
    "type": "object",
    "properties": {
        "contextual_explanation": {"type": "string"},
        "when_this_helps": {"type": "string"},
        "core_principle": {"type": "string"},
        "gujarati_explanation": {"type": "string"},
    },
    "required": ["contextual_explanation", "when_this_helps", "core_principle"],
    "additionalProperties": False,
}


def submit() -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    df = pd.read_parquet(SAMPLE).reset_index(drop=True)
    client = anthropic.Anthropic()
    # Shared system prompt, cached across every request in the batch.
    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]

    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$ — KB ids have dots / are long, so
    # use the row index and keep an index->id map for collection.
    id_map = {f"r{i}": str(row["id"]) for i, row in df.iterrows()}
    requests = [
        Request(
            custom_id=f"r{i}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=1200, system=system,
                messages=[{"role": "user", "content": build_prompt(row.to_dict())}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            ),
        )
        for i, row in df.iterrows()
    ]
    batch = client.messages.batches.create(requests=requests)
    BATCH_META.write_text(json.dumps({"batch_id": batch.id, "n": len(requests),
                                      "id_map": id_map}))
    print(f"submitted batch {batch.id} ({len(requests)} requests) -> {BATCH_META}")
    print("collect with: python -m enrichment.generate_gold collect --batch-id", batch.id)


def collect(batch_id: str | None) -> None:
    import anthropic
    client = anthropic.Anthropic()
    meta = json.loads(BATCH_META.read_text())
    id_map = meta.get("id_map", {})
    if not batch_id:
        batch_id = meta["batch_id"]

    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        print(f"  status={b.processing_status} processing={b.request_counts.processing}")
        time.sleep(30)

    n_ok = n_err = 0
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD, "w") as f:
        for res in client.messages.batches.results(batch_id):
            if res.result.type != "succeeded":
                n_err += 1
                continue
            text = next((b.text for b in res.result.message.content if b.type == "text"), "")
            try:
                data = json.loads(text)
            except Exception:
                n_err += 1
                continue
            rec = {"id": id_map.get(res.custom_id, res.custom_id),
                   **{k: data.get(k) for k in FIELDS}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"wrote {n_ok} gold rows -> {GOLD} (errors: {n_err})")


def retry_missing() -> None:
    """Regenerate rows missing from gold.jsonl (e.g. truncated at max_tokens),
    synchronously with generous headroom, and append them. Cheap (~tens of rows)."""
    import anthropic
    df = pd.read_parquet(SAMPLE)
    df["id"] = df["id"].astype(str)
    have = {json.loads(l)["id"] for l in GOLD.read_text().splitlines()} if GOLD.exists() else set()
    todo = df[~df["id"].isin(have)]
    print(f"missing {len(todo)} rows -> regenerating with max_tokens=1500")
    client = anthropic.Anthropic()
    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    n_ok = n_err = 0
    with open(GOLD, "a") as f:
        for _, row in todo.iterrows():
            try:
                msg = client.messages.create(
                    model=MODEL, max_tokens=1500, system=system,
                    messages=[{"role": "user", "content": build_prompt(row.to_dict())}],
                    output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
                text = next((b.text for b in msg.content if b.type == "text"), "")
                data = json.loads(text)
            except Exception as e:
                n_err += 1
                print(f"  still failed {row['id']}: {type(e).__name__}")
                continue
            rec = {"id": str(row["id"]), **{k: data.get(k) for k in FIELDS}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"appended {n_ok} rows (still failed: {n_err}) -> {GOLD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("submit")
    c = sub.add_parser("collect")
    c.add_argument("--batch-id", default=None)
    sub.add_parser("retry")
    args = ap.parse_args()
    if args.cmd == "submit":
        submit()
    elif args.cmd == "retry":
        retry_missing()
    else:
        collect(args.batch_id)


if __name__ == "__main__":
    main()
