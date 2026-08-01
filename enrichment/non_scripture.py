"""Detect non-scripture front/back matter in the counseling core — by RULE, not by id.

WHY THIS EXISTS (the 2026-08-01 audit finding):

An earlier audit found 14 non-scripture chunks in the core (prefaces, glossaries, an
index, a library circulation slip) and excluded them by listing their ids in
`config/counseling_core.yaml`. Then the KB was re-chunked, every id changed, and 8 of
the 14 exclusions silently stopped binding. The junk came back into the SERVED index:
a conference title page, a list of abbreviations, and an IAST romanisation note were
all retrievable as counseling passages again.

An id list cannot survive a re-chunk. A rule can. This module is evaluated at index
build time, so the exclusion re-derives itself from whatever the corpus currently is.

WHAT IT KEYS ON

Not the source text — front matter looks like ordinary prose to a regex. It keys on the
ENRICHMENT, because the enricher reliably announces what it is looking at ("This is a
table of contents...", "This is a list of abbreviations... not a teaching itself") and
whether it helps ("This does not apply to personal struggles or emotional counseling").
That is a far cleaner signal than anything in the original page.

FALSE POSITIVES THIS DELIBERATELY AVOIDS (each one found by reading the flagged text —
regex detectors are the #1 source of false eval signal on this project):

  * `satsangijivanam_c01869` — "It is not a teaching to be imitated, but a cautionary
    example". A real passage. Hence NEG requires "not a teaching ITSELF", never a bare
    "not a teaching".
  * `chanakya_niti_7.14` — "Circulate wealth to keep it useful" matched an early
    `circulation` pattern aimed at a library slip. Hence the structural patterns are
    ANCHORED to the start of the explanation ("^This is a ...").
  * `bhagavad_gita_8.29` — a chapter colophon, correctly described as "a formal closing
    rather than a teaching". Colophons must stay in the core so exact lookup of
    "Gita 8.29" still resolves; `verse.is_colophon()` already keeps them out of verse
    SEARCH. Hence text_type='verse' is exempt here.

Report what it would drop before trusting it:

    python -m enrichment.non_scripture --report
    python -m enrichment.non_scripture --report --sample 25
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# The enricher naming the row's structural type, anchored at the start of the
# explanation. Anchoring is what keeps "circulate wealth" out of a "circulation" rule.
_STRUCTURAL = re.compile(
    # up to two modifiers, so "a translator's preface" and "an introductory note" match
    r"^this (?:is|passage is|page is) (?:a|an|the) (?:[\w'’-]+ ){0,2}(?:"
    r"title page|list of abbreviations|table of contents|bibliograph\w*|index (?:of|to)|"
    r"glossary|errata|copyright|publication|circulation|catalogu\w*|library|"
    r"technical note|note (?:about|on) how|publisher\w*|preface|foreword|dedication|"
    r"acknowledg\w*|list of (?:committee|organi[sz]ers|patrons|members|contributors)"
    r")",
    re.I,
)

# The enricher explicitly denying this row is a teaching. "itself" is load-bearing:
# without it this matches "not a teaching to be imitated", which is a real passage.
_NOT_A_TEACHING = re.compile(
    r"not a (?:scriptural |spiritual )?(?:teaching|verse) itself"
    r"|not a teaching or verse itself"
    r"|not a teaching verse"
    r"|contains no teaching",
    re.I,
)

# The enricher saying the row helps with nothing — checked on `when_this_helps`, whose
# whole job is to answer "what life situation is this for?".
_NO_HELP = re.compile(
    r"does not (?:offer|provide|apply|help)|no direct help",
    re.I,
)

REASONS = ("structural", "not_a_teaching", "no_help")


def reasons(df: pd.DataFrame) -> pd.Series:
    """Per-row reason string ('' when the row is fine). Vectorised; ~1s over 23k rows."""
    ce = df["contextual_explanation"].fillna("").astype(str)
    wh = df["when_this_helps"].fillna("").astype(str)

    out = pd.Series("", index=df.index, dtype=object)
    out = out.mask(wh.str.contains(_NO_HELP), "no_help")
    out = out.mask(ce.str.contains(_NOT_A_TEACHING), "not_a_teaching")
    out = out.mask(ce.str.contains(_STRUCTURAL), "structural")

    # A colophon is described as "a formal closing rather than a teaching" but must stay
    # addressable for exact lookup, so no verse row is ever dropped by this rule.
    if "text_type" in df.columns:
        out = out.mask(df["text_type"].eq("verse"), "")
    return out


def mask(df: pd.DataFrame) -> pd.Series:
    """True where the row is non-scripture front/back matter and should leave the core."""
    return reasons(df).ne("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="api/data/counseling_index.parquet",
                    help="built index to report against")
    ap.add_argument("--sample", type=int, default=15, help="rows to print per reason")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    df = pd.read_parquet(a.index)
    r = reasons(df)
    hit = r.ne("")
    print(f"{int(hit.sum())} of {len(df)} rows are non-scripture front/back matter "
          f"({hit.mean() * 100:.1f}%)\n")
    print(r[hit].value_counts().to_string())
    print("\nby source (top 12):")
    print(df[hit]["source"].value_counts().head(12).to_string())

    if a.report:
        for reason in REASONS:
            rows = df[r.eq(reason)]
            if rows.empty:
                continue
            print(f"\n--- {reason}: {len(rows)} rows, showing {min(a.sample, len(rows))} ---")
            for _, row in rows.sample(min(a.sample, len(rows)), random_state=0).iterrows():
                exp = " ".join(str(row["contextual_explanation"]).split())[:110]
                print(f"  {row['id'][:44]:44s} | {exp}")


if __name__ == "__main__":
    main()
