"""DPO preference-pair schema for the V2 (Claude-free) generation Gemma.

One pair = a training example: given the same grounded prompt, `chosen` is the
faithful saint-companion reply and `rejected` carries a specific, labelled flaw
(hallucination / sycophancy / doctrine-mixing / shallowness / off-tradition /
name-fabrication) — the exact failure modes V1's evals catch.

The `prompt` we render mirrors V1's real generation user-turn (api/generate.py),
so SFT + DPO train on the same input distribution the model serves at runtime.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from api.generate import _passages_block  # reuse the exact [P#] context format
from api.retrieve_types import Passage

DATA = Path(__file__).resolve().parent / "data"


@dataclass
class PreferencePair:
    problem: str                       # the person's message (user turn)
    context: str                       # the [P#] passages block (grounding)
    chosen: str                        # faithful, grounded saint reply
    rejected: str                      # same prompt, one labelled flaw injected
    flaw: str                          # which failure mode `rejected` carries
    pair_source: str                   # scripture_derived | synthetic | conversation
    grounding_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def render_prompt(self) -> str:
        """The exact user-turn the model is trained/served on (grounded, no reply)."""
        return (f"The person wrote:\n\"{self.problem}\"\n\n"
                f"PASSAGES (cite only these, by tag):\n{self.context}\n\n"
                f"Respond to the person now as the saint-companion.")

    def to_json(self) -> dict:
        return asdict(self)


def context_from_passages(passages: list[Passage]) -> str:
    return _passages_block(passages)


def write_jsonl(pairs: list[PreferencePair], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[PreferencePair]:
    out = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(PreferencePair(**json.loads(line)))
    return out
