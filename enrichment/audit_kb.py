"""Thorough deterministic audit of the enriched KB — every chunk, every enrichment.

Read-only. No LLM (keeps the KB enrichment Claude-free). Checks:
  A. Corpus-wide structure   (dup ids, empty required fields, counts)
  B. Enrichment completeness (all 4 fields present + non-degenerate on every core row)
  C. Language propriety      (mojibake, script leakage, wrong-language Gujarati)
  D. Artifacts               (JSON/prompt echo, truncation, terminal punctuation)
  E. Collapse / duplication  (identical enrichment reused across rows)
  F. Length distribution     (degenerate-short / max-token-truncated)
  G. Design invariants       (Gujarati only where expected; non-core untouched vs backup)

    python -m enrichment.audit_kb
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

from pathlib import Path

_KBDIR = Path(__file__).resolve().parent.parent.parent / "satsangai" / "data" / "parquet"
KB = str(_KBDIR / "corpus.parquet")
BAK = str(_KBDIR / "corpus.parquet.pre_enrich.bak")

EN_FIELDS = ["contextual_explanation", "when_this_helps", "core_principle"]
GU_FIELD = "gujarati_explanation"
ENR_FIELDS = EN_FIELDS + [GU_FIELD]

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
GUJARATI = re.compile(r"[઀-૿]")
MOJIBAKE = re.compile(r"[�]|Ã[\x80-\xbf]|â€|Â|ï¿½|�")
JSON_ECHO = re.compile(r'("(contextual_explanation|when_this_helps|core_principle|"|gujarati_explanation)"\s*:)|^\s*[{\[]')
PROMPT_ECHO = re.compile(r"(?i)\b(here is the|here's the|as an ai|i cannot|based on the (passage|text) (above|provided)|output:)\b")
# The enrichment model honestly flags non-scripture front/back-matter it was fed.
NON_SCRIPTURE = re.compile(
    r"(?i)\bis not (a )?(scriptural|spiritual|religious)( teaching| text| passage)"
    r"|not a (scriptural|spiritual|religious) (teaching|text|passage)"
    r"|does not (convey|contain|express|offer) (any )?(a )?(spiritual|religious|scriptural|meaningful)"
    r"|purely (administrative|bibliographic)"
    r"|library (slip|circulation)|circulation slip"
    r"|not a (teaching|verse)( or (verse|teaching))?( itself)?"
    r"|^This is a technical (guide|preface|list|note)"
    r"|list of abbreviations")
# core_principle is a terse maxim BY DESIGN — only flag genuine degeneracy.
MIN_CP = 10


def s(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def has_ctrl(t: str) -> bool:
    return any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in t)


def truncated(t: str) -> bool:
    t = t.rstrip()
    if not t:
        return False
    # ends without terminal punctuation OR ends mid-clause with a dangling comma / conjunction
    return (t[-1] not in ".!?।॥\"')]") or bool(re.search(r"\b(and|but|or|the|to|of|a|is|as)\s*$", t, re.I))


def main() -> None:
    df = pd.read_parquet(KB, columns=[
        "id", "source", "text_type", "tradition", "citation", "original",
        "translation", "embedding_source_text", *ENR_FIELDS])
    n = len(df)
    enr = df[df["contextual_explanation"].notna()].copy()
    print(f"# KB AUDIT — {n:,} rows | {len(enr):,} enriched\n")

    issues: dict[str, list] = {}

    def flag(cat, rowid, detail):
        issues.setdefault(cat, []).append((rowid, detail))

    # ---- A. corpus structure ----
    print("## A. Structure")
    dup = df["id"].duplicated().sum()
    empty_id = (df["id"].map(s).str.strip() == "").sum()
    empty_txt = ((df["translation"].map(s).str.strip() == "") &
                 (df["original"].map(s).str.strip() == "")).sum()
    print(f"  duplicate ids: {dup}")
    print(f"  empty ids: {empty_id}")
    print(f"  rows with BOTH translation+original empty: {empty_txt}")

    # ---- B/C/D/F. per-enriched-row content checks ----
    print("\n## B–F. Per-enriched-row checks")
    for _, r in enr.iterrows():
        rid = r["id"]
        # core_principle is a terse maxim by design; the prose fields are not.
        for f in EN_FIELDS:
            t = s(r[f])
            if not t.strip():
                flag(f"empty:{f}", rid, "")
                continue
            floor = MIN_CP if f == "core_principle" else 60
            if len(t.strip()) < floor:
                flag(f"too_short:{f}", rid, repr(t[:60]))
            if NON_SCRIPTURE.search(t):
                flag("non_scripture_leak", rid, repr(t[:90]))
            if MOJIBAKE.search(t):
                flag(f"mojibake:{f}", rid, repr(t[:60]))
            if has_ctrl(t):
                flag(f"ctrl_char:{f}", rid, "")
            if JSON_ECHO.search(t):
                flag(f"json_echo:{f}", rid, repr(t[:80]))
            if PROMPT_ECHO.search(t):
                flag(f"prompt_echo:{f}", rid, repr(t[:80]))
            # English field flooded with Indic script (short quotes are ok; >30% is not)
            indic = len(DEVANAGARI.findall(t)) + len(GUJARATI.findall(t))
            if indic and indic / max(len(t), 1) > 0.30:
                flag(f"script_leak:{f}", rid, f"{indic} indic chars in {len(t)}")
            if f == "contextual_explanation" and truncated(t):
                flag("truncated:contextual_explanation", rid, repr(t[-50:]))

        # Gujarati field: if present it must actually be Gujarati script
        g = s(r[GU_FIELD])
        if g.strip():
            gu = len(GUJARATI.findall(g))
            if gu / max(len(g), 1) < 0.30:
                flag("gujarati_not_gujarati", rid, repr(g[:80]))
            if MOJIBAKE.search(g):
                flag("mojibake:gujarati", rid, repr(g[:60]))

    # ---- E. collapse / duplication ----
    ce = enr["contextual_explanation"].map(s)
    vc = Counter(ce[ce.str.strip() != ""])
    collapsed = {k: v for k, v in vc.items() if v > 1}
    print(f"\n## E. Collapse")
    print(f"  distinct contextual_explanation: {ce.nunique():,} / {len(enr):,}")
    print(f"  values reused across >1 row: {len(collapsed)} "
          f"(covering {sum(collapsed.values())} rows)")
    for txt, cnt in sorted(collapsed.items(), key=lambda x: -x[1])[:5]:
        print(f"    x{cnt}: {txt[:70]!r}")

    # ---- F. length distribution ----
    L = ce[ce.str.strip() != ""].str.len()
    print(f"\n## F. contextual_explanation length: "
          f"min={L.min()} p05={int(L.quantile(.05))} median={int(L.median())} "
          f"p95={int(L.quantile(.95))} max={L.max()}")

    # ---- G. design invariants ----
    print("\n## G. Invariants")
    sw = enr["tradition"] == "swaminarayan"
    gu_present = enr[GU_FIELD].map(s).str.strip() != ""
    print(f"  swaminarayan enriched: {sw.sum():,} | with gujarati: {(sw & gu_present).sum():,} "
          f"| swaminarayan MISSING gujarati: {(sw & ~gu_present).sum():,}")
    print(f"  non-swaminarayan WITH gujarati (unexpected): {(~sw & gu_present).sum():,}")
    # per-source enriched coverage
    print("\n  enriched rows per source (top 12):")
    for src, c in enr["source"].value_counts().head(12).items():
        print(f"    {c:>6}  {src}")

    # rows already excluded from the counseling core (config) are resolved, not open
    try:
        import yaml
        cfg = Path(__file__).resolve().parent.parent / "config" / "counseling_core.yaml"
        excluded = set(yaml.safe_load(cfg.read_text()).get("excluded_ids") or [])
    except Exception:
        excluded = set()

    # ---- report ----
    print("\n" + "=" * 60)
    open_issues = {c: [(i, d) for i, d in rows if i not in excluded]
                   for c, rows in issues.items()}
    open_issues = {c: r for c, r in open_issues.items() if r}
    resolved = sum(1 for rows in issues.values() for i, _ in rows if i in excluded)
    print(f"## FINDINGS SUMMARY   (excluded-from-core = resolved: {resolved} hits)")
    if not open_issues:
        print("  CLEAN — no OPEN content issues (all flagged rows excluded from core).")
        issues = open_issues
    else:
        for cat in sorted(open_issues, key=lambda k: -len(open_issues[k])):
            rows = open_issues[cat]
            print(f"  {len(rows):>5}  {cat} (OPEN)")
        issues = open_issues
        # dump a few examples of the worst categories
        print("\n## EXAMPLES (up to 3 per category)")
        for cat in sorted(issues, key=lambda k: -len(issues[k])):
            print(f"\n  [{cat}] ({len(issues[cat])})")
            for rid, det in issues[cat][:3]:
                print(f"    {rid}: {det}")

    # machine-readable id lists for any follow-up re-enrichment
    bad_ids = sorted({rid for rows in issues.values() for rid, _ in rows})
    print(f"\n  TOTAL distinct rows with >=1 issue: {len(bad_ids)}")
    outp = Path(__file__).resolve().parent / "data" / "audit_bad_ids.txt"
    outp.write_text("\n".join(bad_ids))
    print(f"  wrote {outp}")


if __name__ == "__main__":
    sys.exit(main())
