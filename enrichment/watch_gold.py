"""Live monitor for the gold-generation batch + gold.jsonl.

Shows batch progress (succeeded/total) while Claude is generating, then the gold
row count + a couple of sample records once the collector has written the file.

    cd ~/.../satsangAI && source ~/.zshrc && python3 -m enrichment.watch_gold
"""
from __future__ import annotations

import json
import time
from pathlib import Path

META = Path("enrichment/data/gold_batch.json")
GOLD = Path("enrichment/data/gold.jsonl")


def main() -> None:
    import anthropic
    client = anthropic.Anthropic()
    batch_id = json.loads(META.read_text())["batch_id"]
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        ts = time.strftime("%H:%M:%S")
        total = c.succeeded + c.errored + c.processing + c.canceled + c.expired
        line = (f"[{ts}] batch {b.processing_status}: "
                f"{c.succeeded}/{total} done, {c.processing} processing, "
                f"{c.errored} errored")
        if GOLD.exists():
            rows = GOLD.read_text().splitlines()
            print(f"{line}\n  gold.jsonl: {len(rows)} rows written")
            if rows:
                print("  --- sample ---")
                for r in rows[:2]:
                    d = json.loads(r)
                    print(f"  [{d['id']}] {d.get('core_principle')!r} :: "
                          f"{(d.get('contextual_explanation') or '')[:120]}")
            break
        print(line)
        if b.processing_status == "ended":
            print("  batch ended; waiting for collector to write gold.jsonl ...")
        time.sleep(20)


if __name__ == "__main__":
    main()
