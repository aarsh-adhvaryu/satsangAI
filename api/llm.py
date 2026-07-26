"""The single seam through which every UTILITY LLM call passes.

Why this exists: V2's claim is that the shipped system runs without Claude. Generation
already had a Gemma backend, but two Claude calls remained on the hot path of every
turn — `understand()` (Sonnet, plan+emotion) and `extract_facts()` (Haiku, memory). So
`SATSANG_GEN_BACKEND=gemma` produced a Gemma *reply* wrapped in Claude reasoning, and
the system still could not start without an Anthropic key.

Routing both through here means the whole runtime flips with one switch:

    SATSANG_UTILITY_BACKEND=gemma     # plan + memory extraction on Gemma 4
    SATSANG_GEN_BACKEND=gemma         # the saint reply on the tuned adapter
    -> no Anthropic key needed anywhere at runtime

Proposal §10 specifies **Gemma 4 E4B** for the utility role: sub-second, no fine-tuning
needed, and far cheaper to host than pushing planning through the 26B. `SATSANG_UTILITY_MODEL`
overrides it; pointing it at the 26B base works but is wasteful.

Offline evaluation may still use Claude (the judge in eval/ is not runtime) — that
distinction is what keeps the Claude-free claim honest rather than absolute.
"""
from __future__ import annotations

import functools
import json
import re

from . import config


# --------------------------------------------------------------------------- #
#  JSON helpers — small models emit fenced or chatty JSON; repair rather than   #
#  fail, because a planning failure would take down the whole turn.             #
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict | None:
    """Best-effort parse of a JSON object out of a model's free text."""
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


# --------------------------------------------------------------------------- #
#  Claude backend (unchanged behaviour — still the default)                     #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _anthropic():
    import anthropic
    return anthropic.Anthropic()


def _claude_json(system: str, user: str, schema: dict | None, model: str,
                 max_tokens: int) -> dict | None:
    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
    if schema:
        kwargs["tools"] = [{"name": "emit", "description": "Return the result.",
                            "input_schema": schema}]
        kwargs["tool_choice"] = {"type": "tool", "name": "emit"}
    resp = _anthropic().messages.create(**kwargs)
    for b in resp.content:
        if getattr(b, "type", "") == "tool_use":
            return dict(b.input)
    return extract_json("".join(b.text for b in resp.content
                                if getattr(b, "type", "") == "text"))


# --------------------------------------------------------------------------- #
#  Gemma backend — local, no network, no API key                                #
# --------------------------------------------------------------------------- #
def _require_free_vram(need_gb: float = 52.0) -> None:
    """Fail in one second instead of after a 20-minute load that ends in OOM.

    The base is ~52 GB bf16, so exactly ONE process can hold it on an 80 GB card. The
    failure mode without this check is brutal: another job is already holding the GPU,
    this one spends 20 minutes streaming weights off a network disk, and only then dies
    with CUDA OOM — losing the load time and telling you nothing useful.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, _total = torch.cuda.mem_get_info()
        free_gb = free / 1024 ** 3
    except Exception:                                            # noqa: BLE001
        return
    if free_gb < need_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB of GPU memory free; this model needs ~{need_gb:.0f} GB.\n"
            f"Another process is almost certainly still holding it — one 52 GB model fits "
            f"on this card at a time.\n"
            f"  check:  nvidia-smi --query-compute-apps=pid,used_memory --format=csv\n"
            f"  free it: pkill -f 'local_smoke|six_gate|uvicorn'\n"
            f"then re-run this command.")


@functools.lru_cache(maxsize=1)
def _gemma():
    """The utility model. Returns (model, tokenizer, use_base_only).

    Default (`UTILITY_MODEL` unset) REUSES the generation model already resident in VRAM
    and simply disables the LoRA adapter for utility calls — the base answers planning and
    fact-extraction, the tuned adapter answers as the saint. One 52 GB model serves both
    roles, so switching the runtime to Gemma costs no additional VRAM and no download.

    That matters practically: `google/gemma-4-26B-A4B-it` is already cached on this box,
    while the E4B model named in proposal §10 is not. Set `SATSANG_UTILITY_MODEL` to load
    a separate smaller utility model once its identifier is confirmed and pulled.
    """
    import torch
    _require_free_vram()
    if not config.UTILITY_MODEL:
        from .generate import _gemma as _gen_model      # already-loaded base + adapter
        model, tok = _gen_model()
        return model, tok, True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = config.UTILITY_MODEL
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if "26" in name or "A4B" in name:          # MoE base needs the eager-kernel flag
        kw["experts_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model, tok, False


def _gemma_text(system: str, user: str, max_tokens: int) -> str:
    import contextlib

    import torch
    model, tok, use_base_only = _gemma()
    msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    # Planning and fact extraction are NOT saint-persona tasks — the adapter was tuned to
    # answer warmly with [P#] citations, which is wrong for emitting JSON. Run the base.
    ctx = model.disable_adapter() if (use_base_only and hasattr(model, "disable_adapter")) \
        else contextlib.nullcontext()
    with ctx, torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #
def complete_json(system: str, user: str, *, schema: dict | None = None,
                  model: str | None = None, max_tokens: int = 700,
                  fallback: dict | None = None) -> dict:
    """Structured call. Returns `fallback` (or {}) if the model emits unparseable JSON.

    Callers must treat a fallback as 'planning unavailable' and degrade gracefully — a
    malformed plan must never take down a turn, least of all one from someone in distress.
    """
    backend = config.UTILITY_BACKEND
    if backend == "gemma":
        instruction = (f"{system}\n\nReturn ONLY a single JSON object"
                       + (f" matching this schema:\n{json.dumps(schema)}" if schema else "")
                       + ". No prose, no code fence.")
        raw = _gemma_text(instruction, user, max_tokens)
        return extract_json(raw) or (fallback or {})
    got = _claude_json(system, user, schema, model or config.UNDERSTAND_MODEL, max_tokens)
    return got or (fallback or {})


def complete_text(system: str, user: str, *, model: str | None = None,
                  max_tokens: int = 700) -> str:
    """Unstructured call, same routing."""
    if config.UTILITY_BACKEND == "gemma":
        return _gemma_text(system, user, max_tokens)
    resp = _anthropic().messages.create(
        model=model or config.UNDERSTAND_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def runtime_is_claude_free() -> bool:
    """True when no runtime path can reach the Anthropic API."""
    return config.UTILITY_BACKEND == "gemma" and config.GEN_BACKEND == "gemma"
