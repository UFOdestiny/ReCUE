"""Bidirectional doubt probe: contrastive confirm-vs-deny answer stability.

At each step cut of the primary trace, force the answer under THREE cues:
  neutral : "\nThe final answer is \\boxed{"
  confirm : "\nI am confident this is correct.\nThe final answer is \\boxed{"
  deny    : "\nWait, this is wrong. Let me reconsider.\nThe final answer is \\boxed{"
plus capture the neutral forced-answer logprob.

Motivation: a robustly-known answer is INVARIANT to both confirming and denying
pressure; a fragile (wrong) answer is pushed around. The contrastive swing
(confirm vs deny disagreement) is a stronger interventional signal than a single
doubt injection, and is orthogonal to passive answer-convergence.
"""
from __future__ import annotations

import argparse
import json

from acd.env import EXP_ROOT, model_path, save_json
from acd.probe import split_think, segment_steps, cut_points, extract_boxed_head
from acd.generate import build_messages

CUES = {
    "neutral": "\n{ct}\n\nThe final answer is \\boxed{{",
    "confirm": "\nI have checked my work and I am confident it is correct.{ct}\n\nThe final answer is \\boxed{{",
    "deny": "\nWait — I think this is wrong. Let me reconsider carefully.{ct}\n\nThe final answer is \\boxed{{",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--max-probes", type=int, default=6)
    ap.add_argument("--probe-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    if args.limit:
        gen = gen[: args.limit]
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    cue_names = list(CUES.keys())
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
            ot, ct = "<think>", "</think>"
        elif "[THINK]" in ptext:
            ot, ct = "[THINK]", "[/THINK]"
        else:
            ot, ct = None, ""
        is_think = ot is not None
        topen = "" if (ot and base.rstrip().endswith(ot)) else (ot + "\n" if is_think else "")
        segs = segment_steps(split_think(ptext))
        if not segs:
            meta.append({"qi": qi, "cuts": []}); continue
        cuts = cut_points(len(segs), args.max_probes)
        rec = {"qi": qi, "cuts": cuts, "n_segs": len(segs), "start": len(flat)}
        ctag = ct if is_think else ""
        for c in cuts:
            partial = base + topen + "\n\n".join(segs[:c])
            for name in cue_names:
                flat.append(partial + CUES[name].format(ct=ctag))
        meta.append(rec)

    max_in = args.max_model_len - args.probe_tokens - 8
    kept = []
    for p in flat:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            ids = ids[-max_in:]; p = tok.decode(ids)
        kept.append(p)
    print(f"[bidir] {len(gen)} q -> {len(kept)} probes ({len(cue_names)} cues)")

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1)
    outs = llm.generate(kept, sp)

    def lp0(o):
        lps = o.outputs[0].logprobs
        if not lps:
            return None
        tid = o.outputs[0].token_ids[0]
        return lps[0][tid].logprob if tid in lps[0] else None

    nc = len(cue_names)
    results = []
    for rec in meta:
        g = gen[rec["qi"]]
        inter = []
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                row = {"cut": c}
                for j, name in enumerate(cue_names):
                    o = outs[s + k * nc + j]
                    row[name] = extract_boxed_head(o.outputs[0].text)
                    if name == "neutral":
                        row["neutral_lp"] = lp0(o)
                inter.append(row)
        results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                        "n_segs": rec.get("n_segs", 0), "intermediate": inter})

    out = EXP_ROOT / "bidir" / f"{args.gen_tag}_bidir.json"
    save_json(results, out)
    print(f"[bidir] saved -> {out}")


if __name__ == "__main__":
    main()
