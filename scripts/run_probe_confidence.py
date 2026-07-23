"""Lean confidence probe: per-cut neutral answer + forced-answer first-token logprob.

This is the minimal probe needed for the +C (answer-confidence dynamics) signal
across the full model x dataset matrix. One cue only (neutral) -> cheap.
Output schema matches the 'intermediate' list with {cut, neutral, neutral_lp}
so exp.novelty_ablation / analysis can read it directly.
"""
from __future__ import annotations

import argparse
import json

from acd.env import EXP_ROOT, model_path, save_json
from acd.probe import split_think, segment_steps, cut_points, extract_boxed_head
from acd.generate import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--max-probes", type=int, default=8)
    ap.add_argument("--probe-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    args = ap.parse_args()

    out = EXP_ROOT / "conf" / f"{args.gen_tag}_conf.json"
    if out.exists():
        print(f"[conf] exists {out}"); return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    flat, meta = [], []
    for qi, g in enumerate(gen):
        try:
            base = tok.apply_chat_template(build_messages(g["question"], mc=(g.get("type") == "mc")),
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            base = tok.apply_chat_template(build_messages(g["question"], mc=(g.get("type") == "mc")),
                                           tokenize=False, add_generation_prompt=True)
        ptext = g["primary_text"]
        if "<think>" in ptext:
            ot, ct = "<think>", "</think>"
        elif "[THINK]" in ptext:
            ot, ct = "[THINK]", "[/THINK]"
        else:
            ot, ct = None, ""
        is_think = ot is not None
        topen = "" if (ot and base.rstrip().endswith(ot)) else (ot + "\n" if is_think else "")
        elicit = f"\n{ct}\n\nThe final answer is \\boxed{{" if is_think else "\n\nThe final answer is \\boxed{"
        segs = segment_steps(split_think(ptext))
        if not segs:
            meta.append({"qi": qi, "cuts": []}); continue
        cuts = cut_points(len(segs), args.max_probes)
        rec = {"qi": qi, "cuts": cuts, "n_segs": len(segs), "start": len(flat)}
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
    print(f"[conf] {len(gen)} q -> {len(kept)} probes")

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

    results = []
    for rec in meta:
        g = gen[rec["qi"]]
        inter = []
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                o = outs[s + k]
                inter.append({"cut": c, "neutral": extract_boxed_head(o.outputs[0].text),
                              "neutral_lp": lp0(o)})
        results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                        "n_segs": rec.get("n_segs", 0), "intermediate": inter})
    save_json(results, out)
    print(f"[conf] saved -> {out}")


if __name__ == "__main__":
    main()
