"""CPU-only smoke test for the V2 DPO data pipeline (no model, no API, no GPU)."""
from __future__ import annotations

import random

from api.retrieve_types import Passage
from v2 import reject, seeds


def _passage() -> Passage:
    return Passage(
        id="vachanamrut_x", citation="Vachanamrut Gadhada I-1", source="vachanamrut",
        tradition="swaminarayan", score=1.0, original="", translation="",
        contextual_explanation="true peace comes from surrendering the ego to God",
        when_this_helps="It helps when you feel proud, crave recognition, or resent criticism.",
        core_principle="Humility over ego")


def test_first_person_and_templates():
    rng = random.Random(0)
    prob = seeds.to_problem(_passage().when_this_helps, rng)
    assert "you" not in prob.lower().split() and prob.strip()          # became first-person
    assert any(w in prob.lower() for w in ("i ", "i'm", "my", "me"))


def test_every_injector_changes_text_and_labels():
    rng = random.Random(1)
    p = [_passage()]
    chosen = "I hear you. There is a teaching for this. [P1] Be gentle with yourself."
    for fn in reject.INJECTORS:
        res = fn(chosen, p, rng)
        if res is None:
            continue
        rej, flaw = res
        assert rej != chosen and flaw                                  # actually mutated + labelled


def test_hallucination_adds_nonexistent_tag():
    rng = random.Random(2)
    p = [_passage()]
    rej, flaw = reject.hallucinate_citation("Grounded reply. [P1]", p, rng)
    assert flaw == "hallucinated_citation" and "[P2]" in rej           # a tag beyond the 1 passage


def test_doctrine_mix_only_for_swaminarayan():
    rng = random.Random(3)
    shared = Passage(id="g", citation="Gita 2.47", source="bhagavad_gita",
                     tradition="shared_hindu", score=1.0, original="", translation="",
                     contextual_explanation="act without attachment to results",
                     when_this_helps="when anxious about outcomes", core_principle="detach")
    assert reject.doctrine_mix("x [P1]", [shared], rng) is None        # not applied off-home
    assert reject.doctrine_mix("x [P1]", [_passage()], rng) is not None


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
