"""Scripture-derived DPO problem seeds — mined from the enriched core, Claude-free.

Each enriched row's `when_this_helps` describes the life situation a passage
addresses ("It helps when you feel pride, crave recognition..."). We turn that
into a first-person problem statement and pair it with the row as grounding —
giving realistic (problem, grounding-passage) seeds without any model or API.
Diversity comes from stratified sampling across source × core_principle.
"""
from __future__ import annotations

import random
import re

import pandas as pd

from api.retrieve_types import Passage

# strip any intro clause up to the connector after a help-word:
# "It helps when", "This is most helpful when", "Helpful for", "Useful during" ...
_LEAD = re.compile(
    r"(?i)^.*?\b(help\w*|useful|valuable|comforting|relevant)\b"
    r"(?:\s+\w+){0,3}?\s+(when|for|if|to|during|in|while)\s+")
# third-person subjects some rows use ("someone feels", "a person is") -> first person.
# NB: no bare "one" — it wrongly matches "loved one", "no one", "each one".
_SUBJ = [(r"\b(someone|a person|a devotee|people|they)\b\s+", "I ", re.I)]
_YOU = [(r"\byou are\b", "I am"), (r"\byou're\b", "I'm"), (r"\byou feel\b", "I feel"),
        (r"\byourself\b", "myself"), (r"\byour\b", "my"), (r"\byou\b", "I")]
# 3rd-person-singular -> 1st-person verb agreement for the verbs common in this field
_VERBS = ["feel", "want", "crave", "seek", "struggle", "question", "doubt", "resent",
          "need", "fear", "face", "carry", "lack", "long", "wish", "wonder"]
_AGREE = [(re.compile(rf"\bI {v}s\b"), f"I {v}") for v in _VERBS] + [
    (re.compile(r"\bI is\b"), "I am"), (re.compile(r"\bI has\b"), "I have"),
    (re.compile(r"\bI does\b"), "I do"), (re.compile(r"\bI was\b"), "I was")]

_TEMPLATES = [
    "Lately {sit}. I don't know how to deal with it.",
    "{Sit_cap}, and it's been weighing on me. What should I do?",
    "I've been struggling — {sit}. Can you help me?",
    "Honestly {sit}. I feel stuck.",
    "{Sit_cap}. How do I find some peace with this?",
]


def _to_first_person(when_this_helps: str) -> str:
    s = _LEAD.sub("", str(when_this_helps).strip()).rstrip(". ")
    for pat, repl, flags in _SUBJ:
        s = re.sub(pat, repl, s, flags=flags)
    for pat, repl in _YOU:
        s = re.sub(pat, repl, s, flags=re.I)
    for pat, repl in _AGREE:                        # fix "I feels" -> "I feel", etc.
        s = pat.sub(repl, s)
    # object-position pronoun: "understands I" -> "understands me" (transitive verbs)
    s = re.sub(r"\b(understand|help|guide|support|comfort|love|hurt|betray|abandon|teach|"
               r"remind|push|challenge|judge|criticize|attack|surround|overwhelm)(s|es)?\s+I\b",
               r"\1\2 me", s, flags=re.I)
    s = re.sub(r"\btheir\b", "my", s, flags=re.I)
    s = re.sub(r"\bone's\b", "my", s, flags=re.I)
    s = re.sub(r"(^|\s)i(\s|$|,|\.)", r"\1I\2", s)   # capitalize standalone pronoun "i"
    return re.sub(r"\s{2,}", " ", s).strip()


def to_problem(when_this_helps: str, rng: random.Random) -> str:
    sit = _to_first_person(when_this_helps)
    if not sit:
        return "I'm going through something difficult and I don't know where to turn."
    # mid-sentence keep a leading pronoun "I" capital, else lowercase; sentence-start upper
    mid = sit if re.match(r"I($|[\s,])", sit) else sit[0].lower() + sit[1:]
    cap = sit[0].upper() + sit[1:]
    return rng.choice(_TEMPLATES).format(sit=mid, Sit_cap=cap)


def _passage_from_row(row: pd.Series) -> Passage:
    return Passage.from_row(row.to_dict(), float(row.get("score", 1.0)))


def sample_seeds(index_parquet: str, n: int, seed: int = 0):
    """Return [(problem, [grounding Passage], row_id)] stratified over source×principle."""
    rng = random.Random(seed)
    df = pd.read_parquet(index_parquet)
    df = df[df["when_this_helps"].notna() & (df["when_this_helps"].str.len() > 20)]
    # stratify: proportional-ish per source, shuffled within
    picks = []
    for _, grp in df.groupby("source"):
        k = max(1, round(n * len(grp) / len(df)))
        picks.append(grp.sample(min(k, len(grp)), random_state=seed))
    pool = pd.concat(picks).sample(frac=1, random_state=seed).head(n)
    out = []
    for _, row in pool.iterrows():
        out.append((to_problem(row["when_this_helps"], rng),
                    [_passage_from_row(row)], row["id"]))
    return out
