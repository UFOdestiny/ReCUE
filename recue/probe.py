"""Intermediate-answer stabilization probe (single-generation, judge-free).

For each cached primary reasoning chain, cut the <think> trace at reasoning-step
boundaries and force-decode "The final answer is \\boxed{...}" at each cut.
This reuses the shared prefix via vLLM automatic prefix caching, so the marginal
cost per probe is ~a dozen decoded tokens, NOT a full re-generation.

Outputs, per question, the ordered list of intermediate answers -> downstream
dynamics features (convergence step, flip count, agreement fraction, entropy).
"""
from __future__ import annotations

import argparse
import re
import json
from pathlib import Path
from typing import List, Optional

from recue.env import EXP_ROOT, model_path, save_json, extract_pred_math, normalize_num


ELICIT = "\n</think>\n\nThe final answer is \\boxed{"


def split_think(primary_text: str) -> str:
    """Return the reasoning body. Handles <think></think> and [THINK][/THINK]."""
    t = primary_text
    for open_tag, close_tag in (("<think>", "</think>"), ("[THINK]", "[/THINK]")):
        if open_tag in t:
            t = t.split(open_tag, 1)[1]
            if close_tag in t:
                t = t.split(close_tag, 1)[0]
            return t.strip("\n")
    return t.strip("\n")


def segment_steps(think_body: str) -> List[str]:
    """Split a thinking trace into ordered reasoning segments."""
    # primary split on blank lines; keeps paragraph-level steps
    segs = [s.strip() for s in re.split(r"\n\s*\n", think_body) if s.strip()]
    if len(segs) <= 1:
        # fallback: split on sentence-ish boundaries
        segs = [s.strip() for s in re.split(r"(?<=[.!?])\s+", think_body) if s.strip()]
    return segs


def cut_points(n_segs: int, max_probes: int) -> List[int]:
    """Choose up to max_probes cut indices (number of segments kept), 1..n_segs."""
    if n_segs <= max_probes:
        return list(range(1, n_segs + 1))
    # evenly spaced fractions, always include the last (full) cut
    import numpy as np
    fr = np.linspace(1.0 / max_probes, 1.0, max_probes)
    idx = sorted(set(int(round(f * n_segs)) for f in fr))
    idx = [max(1, min(n_segs, i)) for i in idx]
    return sorted(set(idx))


def extract_boxed_head(text: str) -> Optional[str]:
    """The generation begins right after '\\boxed{'; read until closing brace."""
    depth = 1
    out = []
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    inner = "".join(out).strip()
    n = normalize_num(inner)
    return n if n is not None else (inner if inner else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--gen-tag", required=True, help="tag of cached gen file")
    ap.add_argument("--max-probes", type=int, default=8)
    ap.add_argument("--probe-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    if args.limit:
        gen = gen[: args.limit]
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    from recue.generate import build_messages

    # Build all probe prompts (flat), remembering (qidx, cut_rank, n_cuts)
    flat_prompts = []
    meta = []
    for qi, r in enumerate(gen):
        try:
            base = tok.apply_chat_template(build_messages(r["question"], mc=(r.get("type") == "mc")),
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            base = tok.apply_chat_template(build_messages(r["question"], mc=(r.get("type") == "mc")),
                                           tokenize=False, add_generation_prompt=True)
        # detect thinking marker style from the actual generation
        ptext = r["primary_text"]
        if "<think>" in ptext:
            open_tag, close_tag = "<think>", "</think>"
        elif "[THINK]" in ptext:
            open_tag, close_tag = "[THINK]", "[/THINK]"
        else:
            open_tag, close_tag = None, None
        is_think = open_tag is not None
        think = split_think(ptext)
        segs = segment_steps(think)
        if not segs:
            meta.append({"qi": qi, "cuts": []})
            continue
        cuts = cut_points(len(segs), args.max_probes)
        rec = {"qi": qi, "cuts": cuts, "n_segs": len(segs), "start": len(flat_prompts)}
        # elicitation tail respects the model's thinking-close marker
        elicit = f"\n{close_tag}\n\nThe final answer is \\boxed{{" if is_think else "\n\nThe final answer is \\boxed{"
        think_open = "" if (open_tag and base.rstrip().endswith(open_tag)) else (open_tag + "\n" if is_think else "")
        for c in cuts:
            partial = "\n\n".join(segs[:c])
            if is_think:
                prompt = base + think_open + partial + elicit
            else:
                # non-thinking model: partial CoT then force the answer line
                prompt = base + partial + "\n\nThe final answer is \\boxed{"
            flat_prompts.append(prompt)
        meta.append(rec)

    print(f"[probe] {len(gen)} questions -> {len(flat_prompts)} probe prompts")

    # guard: truncate probe prompts that would overflow the context window
    max_in = args.max_model_len - args.probe_tokens - 8
    kept = []
    for p in flat_prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            # keep head (system+question+think start) impossible to reconstruct cheaply;
            # instead drop the last tokens of the think body but preserve the ELICIT tail
            elicit_ids = tok(ELICIT, add_special_tokens=False)["input_ids"]
            body = ids[: max_in - len(elicit_ids)]
            ids = body + elicit_ids
            p = tok.decode(ids)
        kept.append(p)
    flat_prompts = kept

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True, enforce_eager=False)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1)
    outs = llm.generate(flat_prompts, sp)

    probe_texts = [o.outputs[0].text for o in outs]
    # also token count for cost accounting
    probe_ntok = [len(o.outputs[0].token_ids) for o in outs]

    results = []
    for rec in meta:
        qi = rec["qi"]
        r = gen[qi]
        final_ans = extract_pred_math(r["primary_text"])
        inter = []
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                a = extract_boxed_head(probe_texts[s + k])
                inter.append({"cut": c, "answer": a})
        results.append({
            "id": r["id"], "gold": r["gold"], "gold_raw": r.get("gold_raw"),
            "final_answer": final_ans,
            "n_segs": rec.get("n_segs", 0),
            "intermediate": inter,
            "probe_ntok": [probe_ntok[rec["start"] + k] for k in range(len(rec.get("cuts", [])))] if rec.get("cuts") else [],
        })

    out_path = EXP_ROOT / "probe" / f"{args.gen_tag}_probe.json"
    save_json(results, out_path)
    print(f"[probe] saved -> {out_path}")


if __name__ == "__main__":
    main()
