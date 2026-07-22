"""Per-sample intra-trace stability probe (for within-problem AUROC ablation).

Unlike exp.probe (which probes only the primary trace), this probes EACH of the
k sampled traces of a problem, but ONLY for problems that have both correct and
wrong traces (the only ones that contribute to within-problem AUROC). This keeps
cost bounded while enabling the decisive difficulty-controlled test with OUR
actual stability signal (not just the agreement proxy).
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from acd.env import EXP_ROOT, model_path, save_json, extract_pred_math
from acd.probe import split_think, segment_steps, cut_points, extract_boxed_head
from acd.generate import build_messages
from acd import data as dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--max-probes", type=int, default=6)
    ap.add_argument("--probe-tokens", type=int, default=16)
    ap.add_argument("--max-problems", type=int, default=150)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    # keep only mixed problems (both correct & wrong among samples)
    mixed = []
    for g in gen:
        cs = []
        for s in g.get("samples", []):
            cs.append(dv.verify({"gold": g["gold"], "gold_raw": g.get("gold_raw")}, s))
        cs = np.array(cs)
        if cs.sum() > 0 and (1 - cs).sum() > 0:
            mixed.append((g, cs))
    if len(mixed) > args.max_problems:
        # deterministic subsample
        idx = np.linspace(0, len(mixed) - 1, args.max_problems).astype(int)
        mixed = [mixed[i] for i in idx]
    print(f"[psample] {len(mixed)} mixed problems from {len(gen)}")

    flat, meta = [], []
    for pi, (g, cs) in enumerate(mixed):
        try:
            base = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            base = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                           add_generation_prompt=True)
        for si, samp in enumerate(g["samples"]):
            if "<think>" in samp:
                ot, ct = "<think>", "</think>"
            elif "[THINK]" in samp:
                ot, ct = "[THINK]", "[/THINK]"
            else:
                ot, ct = None, None
            is_think = ot is not None
            segs = segment_steps(split_think(samp))
            if not segs:
                meta.append({"pi": pi, "si": si, "cuts": [], "corr": int(cs[si])}); continue
            cuts = cut_points(len(segs), args.max_probes)
            elicit = f"\n{ct}\n\nThe final answer is \\boxed{{" if is_think else "\n\nThe final answer is \\boxed{"
            topen = "" if (ot and base.rstrip().endswith(ot)) else (ot + "\n" if is_think else "")
            rec = {"pi": pi, "si": si, "cuts": cuts, "n_segs": len(segs),
                   "start": len(flat), "corr": int(cs[si])}
            for c in cuts:
                flat.append(base + topen + "\n\n".join(segs[:c]) + elicit)
            meta.append(rec)

    # truncate overlong
    max_in = args.max_model_len - args.probe_tokens - 8
    kept = []
    for p in flat:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            ids = ids[-max_in:]; p = tok.decode(ids)
        kept.append(p)
    print(f"[psample] {len(kept)} probes")

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1)
    outs = llm.generate(kept, sp)
    texts = [o.outputs[0].text for o in outs]

    # assemble: per (problem, sample) intermediate answers + final answer
    results = []
    for rec in meta:
        g = mixed[rec["pi"]][0]
        samp = g["samples"][rec["si"]]
        final = extract_pred_math(samp)
        inter = []
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                inter.append({"cut": c, "answer": extract_boxed_head(texts[s + k])})
        results.append({"pid": g["id"], "si": rec["si"], "corr": rec["corr"],
                        "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                        "final_answer": final, "n_segs": rec.get("n_segs", 0),
                        "intermediate": inter})

    out = EXP_ROOT / "persample" / f"{args.gen_tag}_ps.json"
    save_json(results, out)
    print(f"[psample] saved -> {out}")


if __name__ == "__main__":
    main()
