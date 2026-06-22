"""Topic-switch eval — can one session handle the user changing topics and asking
different KINDS of questions (emotional problems AND a factual/teaching question)?

Runs multi-turn sessions over a shared conversation_id with deliberate switches, then
an Opus judge checks, per switched turn: did the response focus on the CURRENT topic
(not stay stuck on the previous one), handle the switch coherently, and retrieve
passages relevant to the new topic?

    source ~/.zshrc && HF_HUB_OFFLINE=1 python -m eval.topic_switch
"""
from __future__ import annotations

import functools
import json
import uuid

from api.pipeline import respond

SESSIONS = [
    {"name": "emotional->emotional->factual->emotional", "turns": [
        ("I'm so angry at my father — he dismisses everything I say.", "anger at father"),
        ("Let's leave that. Honestly I'm more scared about losing my job right now.", "fear of job loss"),
        ("By the way, what does the Gita actually teach about doing one's duty without "
         "attachment?", "a teaching question about duty/detachment"),
        ("I don't know... lately I just feel empty and pointless.", "emptiness / lack of meaning"),
    ]},
    {"name": "rapid different questions", "turns": [
        ("How do I find peace when my mind won't stop racing?", "restlessness of mind"),
        ("Different question — is it wrong to feel jealous of my brother?", "jealousy of a sibling"),
        ("And how do I forgive someone who won't even apologize?", "forgiving the unrepentant"),
    ]},
]

SCHEMA = {"type": "object", "properties": {
    "on_topic": {"type": "boolean"}, "coherent_switch": {"type": "boolean"},
    "retrieval_relevant": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["on_topic", "coherent_switch", "retrieval_relevant", "reason"],
    "additionalProperties": False}


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def _judge(prev_topic, cur_topic, message, citations, reply):
    content = (f"This is turn N of a single conversation. The PREVIOUS turn was about: "
               f"{prev_topic}. The user's CURRENT message is about: {cur_topic}.\n\n"
               f"CURRENT message: \"{message}\"\nRetrieved passage citations: {citations}\n"
               f"SYSTEM RESPONSE:\n{reply}\n\n"
               f"Judge: on_topic = the response addresses the CURRENT topic, not stuck on "
               f"the previous one. coherent_switch = it handles the change of subject "
               f"naturally (doesn't act confused or force the old topic). "
               f"retrieval_relevant = the cited passages relate to the CURRENT topic.")
    r = _client().messages.create(model="claude-opus-4-8", max_tokens=400,
        system="You strictly evaluate whether a counseling companion follows topic "
               "changes within one conversation. Return strict JSON.",
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    return json.loads(next(b.text for b in r.content if b.type == "text"))


def main() -> None:
    passed = total = 0
    for sess in SESSIONS:
        cid = f"topic-{uuid.uuid4().hex[:8]}"
        print(f"\n=== session: {sess['name']} ===")
        prev_topic = None
        for i, (msg, topic) in enumerate(sess["turns"]):
            reply, cites = [], []
            for ev, pl in respond(msg, conversation_id=cid, user_id=None):
                if ev == "text":
                    reply.append(pl)
                elif ev == "passages":
                    cites = [p["citation"] for p in pl]
            reply = "".join(reply)
            if i == 0:                       # first turn sets the topic; nothing to switch
                prev_topic = topic
                print(f"  T{i+1} [{topic}] (baseline)")
                continue
            j = _judge(prev_topic, topic, msg, cites, reply)
            ok = j["on_topic"] and j["coherent_switch"] and j["retrieval_relevant"]
            passed += ok
            total += 1
            print(f"  T{i+1} [{topic}] {'✅' if ok else '❌'} "
                  f"on_topic={j['on_topic']} coherent={j['coherent_switch']} "
                  f"retr_rel={j['retrieval_relevant']}")
            if not ok:
                print(f"        reason: {j['reason'][:160]}")
            prev_topic = topic
    print(f"\n=== TOPIC-SWITCH PASS: {passed}/{total} switched turns ===")


if __name__ == "__main__":
    main()
