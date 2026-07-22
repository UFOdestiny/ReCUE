"""Per-sample confidence-dynamics probe (for confidence-aware voting).

For EACH of the k sampled traces of a question, cut at a few step boundaries and
force-decode the answer + first-token logprob. Yields, per sample: its answer and
a confidence-dynamics summary (final conf, mean conf, slope, min). Enables
'confidence-dynamics-weighted self-consistency' at matched k-sample budget.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from acd.env import EXP_ROOT, model_path, save_json, extract_pred_math
from acd.probe import split_think, segment_steps, cut_points, extract_boxed_head
from acd.generate import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cuts", type=int, default=4)
    ap.add_argument("--probe-tokens", type=int, default=12)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = EXP_ROOT / "sampleconf" / f"{args.gen_tag}_sc.json"
    if out.exists():
        print(f"[sampleconf] exists {out}"); return

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
        for si, samp in enumerate(g["samples"][:args.k]):
            if "<think>" in samp: ot, ct = "<think>", "</think>"
            elif "[THINK]" in samp: ot, ct = "[THINK]", "[/THINK]"
            else: ot, ct = None, ""
            is_think = ot is not None
            topen = "" if (ot and base.rstrip().endswith(ot)) else (ot + "\n" if is_think else "")
            elicit = f"\n{ct}\n\nThe final answer is \\boxed{{" if is_think else "\n\nThe final answer is \\boxed{"
            segs = segment_steps(split_think(samp))
            if not segs:
                meta.append({"qi": qi, "si": si, "cuts": [], "final": extract_pred_math(samp)}); continue
            cuts = cut_points(len(segs), args.cuts)
            rec = {"qi": qi, "si": si, "cuts": cuts, "start": len(flat), "final": extract_pred_math(samp)}
            for c in cuts:
                flat.append(base + topen + "\n\n".join(segs[:c]) + elicit)
            meta.append(rec)

    max_in = args.max_model_len - args.probe_tokens - 8
    kept = []
    for p in flat:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            ids = ids[-max_in:]; p = tok.decode(ids)
        kept.append(p)
    print(f"[sampleconf] {len(gen)} q x {args.k} samples -> {len(kept)} probes")

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1)
    outs = llm.generate(kept, sp)

    def lp0(o):
        lps = o.outputs[0].logprobs
        if not lps: return None
        tid = o.outputs[0].token_ids[0]
        return lps[0][tid].logprob if tid in lps[0] else None

    # group by question -> list of per-sample dynamics
    by_q = {}
    for rec in meta:
        qi = rec["qi"]
        entry = {"si": rec["si"], "final": rec["final"], "inter": []}
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                o = outs[s + k]
                entry["inter"].append({"cut": c, "answer": extract_boxed_head(o.outputs[0].text),
                                       "lp": lp0(o)})
        by_q.setdefault(qi, []).append(entry)

    results = []
    for qi, g in enumerate(gen):
        results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                        "samples": by_q.get(qi, [])})
    save_json(results, out)
    print(f"[sampleconf] saved -> {out}")


if __name__ == "__main__":
    main()
