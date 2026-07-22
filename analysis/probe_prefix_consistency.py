"""Prefix Consistency baseline (Reliable CoT via Prefix Consistency, 2026).

Procedure: cut the primary reasoning trace at a few prefix points; from each
prefix, RESAMPLE N continuations (temperature>0) that finish the reasoning and
give an answer; extract each continuation's answer. The per-response score is the
reproducibility/consistency of the resampled answers with the final answer
(higher => more reliable). Unlike our probe (temp=0 forced single-token answer,
reuses cache), this genuinely re-generates continuations -> much more expensive.

Score variants stored per response:
  pc_reproduce  : mean agreement of resampled-continuation answers with final answer
  pc_selfcons   : mean self-consistency (modal-answer fraction) across all resampled answers
  pc_lastprefix : reproduce fraction at the LAST prefix (deepest, most info)
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from collections import Counter

from acd.env import EXP_ROOT, model_path, save_json, extract_pred_math
from acd.probe import split_think, segment_steps, cut_points
from acd.generate import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--n-prefix", type=int, default=3, help="prefix cut points")
    ap.add_argument("--n-cont", type=int, default=4, help="continuations resampled per prefix")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = EXP_ROOT / "prefix" / f"{args.gen_tag}_prefix.json"
    if out.exists():
        print(f"[prefix] exists {out}"); return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    if args.limit:
        gen = gen[: args.limit]
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    flat, meta = [], []
    for qi, g in enumerate(gen):
        try:
            base = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            base = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                           add_generation_prompt=True)
        ptext = g["primary_text"]
        if "<think>" in ptext:
            ot = "<think>"
        elif "[THINK]" in ptext:
            ot = "[THINK]"
        else:
            ot = None
        topen = "" if (ot and base.rstrip().endswith(ot)) else (ot + "\n" if ot else "")
        segs = segment_steps(split_think(ptext))
        if not segs:
            meta.append({"qi": qi, "cuts": []}); continue
        # prefix cut points: earlier fractions (prefix reproducibility) — use first n_prefix
        cuts = cut_points(len(segs), args.n_prefix + 1)[:args.n_prefix]
        rec = {"qi": qi, "cuts": cuts, "start": len(flat)}
        for c in cuts:
            # prefix = partial reasoning; let the model continue freely
            flat.append(base + topen + "\n\n".join(segs[:c]) + "\n")
        meta.append(rec)

    max_in = args.max_model_len - args.max_new - 8
    kept = []
    for p in flat:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            ids = ids[-max_in:]; p = tok.decode(ids)
        kept.append(p)
    print(f"[prefix] {len(gen)} q -> {len(kept)} prefixes x n_cont={args.n_cont}")

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    sp = SamplingParams(n=args.n_cont, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_new, seed=123)
    outs = llm.generate(kept, sp)

    results = []
    for rec in meta:
        g = gen[rec["qi"]]
        final = extract_pred_math(g["primary_text"])
        prefixes = []
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                o = outs[s + k]
                answers = [extract_pred_math(comp.text) for comp in o.outputs]
                prefixes.append({"cut": c, "answers": answers})
        results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                        "final_answer": final, "prefixes": prefixes})
    save_json(results, out)
    print(f"[prefix] saved -> {out}")


if __name__ == "__main__":
    main()
