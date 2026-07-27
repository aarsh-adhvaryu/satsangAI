"""Quantize the merged V2 model so it fits commodity GPUs (proposal §21).

The economics, which are the whole point:

    bf16   51.6 GB  -> A100-80 / H100 only     ~$2,880/mo always-on on Lightning
    8-bit  ~26 GB   -> L40S, A100-40
    4-bit  ~13 GB   -> T4 16GB, L4, RTX 4090   ~$250-320/mo, or ~$30 scale-to-zero

25.8B params, MoE with ~4B active per token. At 4-bit the model stops needing datacentre
hardware, which is what makes self-hosting an open model cheaper than renting an API.

    # 1. merge the LoRA into the base (GPU, writes ~52 GB)
    python -m v2.serve_vllm merge --adapter v2/data/gemma4-v2-dpo2-lora \\
        --out v2/data/gemma4-v2-merged
    # 2. quantize (GPU)
    python -m v2.quantize --model v2/data/gemma4-v2-merged --out v2/data/gemma4-v2-4bit
    # 3. PROVE IT — 99-probe gate against the bf16 baseline, no API cost
    #    (see v2/quantize.py --help for the exact eval command)

QUALITY IS NOT ASSUMED. 4-bit on a DPO-tuned MoE is not well-trodden: quantization error
can land unevenly across experts, and this adapter has been through both SFT and DPO. The
deterministic gate is the arbiter — `eval/six_gate_final.json` is the bf16 baseline
(counseling/verse/creative/out_of_domain all 1.000, routing 34/34, 0 breaches). If 4-bit
holds those, ship it; if not, fall back to 8-bit at ~26 GB and re-measure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CALIB_PROMPTS = [
    # Calibration should look like SERVED traffic, not generic web text: the same
    # grounded, passage-citing, saint-persona distribution the adapter was tuned on.
    "The person wrote:\n\"I keep losing my temper with my mother and I feel terrible after\"\n\n"
    "PASSAGES (cite only these, by tag):\n[P1] Vachanamrut Gadhada I-58\n  text: anger arises "
    "from unfulfilled desire\n  meaning: when what we want is blocked, anger follows\n\n"
    "Respond to the person now as the saint-companion.",
    "The person wrote:\n\"What is the difference between atma and jiva?\"\n\n"
    "PASSAGES (cite only these, by tag):\n[P1] Satsang Reader glossary\n  text: atma — the "
    "conscious eternal self\n  meaning: the self distinct from body and mind\n\n"
    "Respond to the person now as the saint-companion.",
    "The person wrote:\n\"મને મારા જીવનમાં કોઈ અર્થ દેખાતો નથી\"\n\n"
    "PASSAGES (cite only these, by tag):\n[P1] Swamini Vato 5/232\n  text: remembrance of God "
    "steadies the restless mind\n  meaning: turning to God gives the mind somewhere to rest\n\n"
    "Respond to the person now as the saint-companion.",
    "The person wrote:\n\"Write me a short poem in English about letting go\"\n\n"
    "PASSAGES (cite only these, by tag):\n[P1] Bhagavad Gita 2.47\n  text: you have the right "
    "to work alone, but not to the fruits of it\n  meaning: act without clinging to outcome\n\n"
    "Respond to the person now as the saint-companion.",
]


def quantize(model: str, out: str, bits: int, method: str, calib: int) -> None:
    src, dst = Path(model), Path(out)
    if not src.exists():
        raise SystemExit(f"merged model not found: {src}\n"
                         f"run:  python -m v2.serve_vllm merge --adapter "
                         f"v2/data/gemma4-v2-dpo2-lora --out {src}")
    dst.mkdir(parents=True, exist_ok=True)

    if method == "bnb":
        # bitsandbytes: no calibration pass, works with plain transformers, and is the
        # safest first attempt on an MoE. NOT servable by vLLM — use it to answer the
        # quality question quickly, then redo with awq/gptq if the gates hold.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        qc = BitsAndBytesConfig(
            load_in_4bit=(bits == 4), load_in_8bit=(bits == 8),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(str(src))
        m = AutoModelForCausalLM.from_pretrained(
            str(src), quantization_config=qc, device_map="auto",
            experts_implementation="eager")     # Gemma-4 MoE: see CLAUDE.md gotchas
        m.save_pretrained(str(dst))
        tok.save_pretrained(str(dst))
    elif method == "awq":
        # AWQ: activation-aware, vLLM-servable. Needs a calibration pass over prompts that
        # resemble real traffic — generic web text calibrates the wrong activations.
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(src))
        m = AutoAWQForCausalLM.from_pretrained(str(src), device_map="auto")
        m.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128,
                                      "w_bit": bits, "version": "GEMM"},
                   calib_data=(CALIB_PROMPTS * ((calib // len(CALIB_PROMPTS)) + 1))[:calib])
        m.save_quantized(str(dst))
        tok.save_pretrained(str(dst))
    else:
        raise SystemExit(f"unknown method: {method}")

    total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"\nwrote {dst}  ({total / 1e9:.1f} GB, was 51.6 GB bf16)")
    print("\nNOW PROVE IT — 99 probes, no API, against the bf16 baseline:")
    print(f"   env -u ANTHROPIC_API_KEY SATSANG_UTILITY_BACKEND=gemma "
          f"SATSANG_GEN_BACKEND=gemma \\\n"
          f"     SATSANG_GEMMA_ADAPTER={dst} SATSANG_EMBED_DEVICE=cuda HF_HUB_OFFLINE=1 \\\n"
          f"     python -m eval.six_gate --backend claude --judge none --k 1 "
          f"--out eval/six_gate_4bit.json")
    print("   python -m v2.compare_gates eval/six_gate_final.json eval/six_gate_4bit.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="v2/data/gemma4-v2-merged",
                    help="MERGED model (not the adapter) — run serve_vllm merge first")
    ap.add_argument("--out", default="v2/data/gemma4-v2-4bit")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--method", default="bnb", choices=["bnb", "awq"],
                    help="bnb = fastest answer to 'does quality hold'; "
                         "awq = what vLLM can actually serve")
    ap.add_argument("--calib", type=int, default=64, help="awq calibration samples")
    a = ap.parse_args()
    quantize(a.model, a.out, a.bits, a.method, a.calib)


if __name__ == "__main__":
    main()
