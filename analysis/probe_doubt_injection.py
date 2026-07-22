"""Novelty probe: counterfactual doubt-injection robustness + forced-answer confidence.

For each cached primary chain, at the same step-boundary cuts as exp.probe, we
force an answer under TWO conditions:
  - neutral   : "...\nThe final answer is \\boxed{"           (baseline commit)
  - challenge : inject a doubt cue before eliciting, e.g.
                "\nWait, I think I may have made a mistake. Let me reconsider.\n
                 The final answer is \\boxed{"

Signals derived downstream:
  - challenge_flip_rate : fraction of cuts where challenge answer != neutral answer
                          (robustness to injected doubt; motivation: correct
                          reasoning is robust, wrong reasoning capitulates)
  - answer_logprob_traj : logprob the model assigns to the FIRST answer token at
                          each neutral cut -> trend/mean/last (commitment strength)

We also record the neutral answers so this file is self-contained.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from acd.env import EXP_ROOT, model_path, save_json, normalize_num
from acd.probe import split_think, segment_steps, cut_points, extract_boxed_head
from acd.generate import build_messages

NEUTRAL = "\nThe final answer is \\boxed{"
CHALLENGE = ("\nWait, I think I may have made a mistake somewhere. Let me reconsider "
             "carefully before committing.\nThe final answer is \\boxed{")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--max-probes", type=int, default=6)
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

    flat, meta = [], []
    for qi, g in enumerate(gen):
        try:
            base = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            base = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                           add_generation_prompt=True)
        is_think = "<think>" in g["primary_text"]
        think_open = "" if base.rstrip().endswith("<think>") else ("<think>\n" if is_think else "")
        segs = segment_steps(split_think(g["primary_text"]))
        if not segs:
            meta.append({"qi": qi, "cuts": []}); continue
        cuts = cut_points(len(segs), args.max_probes)
        rec = {"qi": qi, "cuts": cuts, "n_segs": len(segs), "start": len(flat)}
        for c in cuts:
            partial = base + think_open + "\n\n".join(segs[:c])
            flat.append(partial + NEUTRAL)      # neutral
            flat.append(partial + CHALLENGE)    # challenge
        meta.append(rec)

    # truncate overlong
    max_in = args.max_model_len - args.probe_tokens - 8
    kept = []
    for p in flat:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            ids = ids[-max_in:]
            p = tok.decode(ids)
        kept.append(p)

    print(f"[chal] {len(gen)} q -> {len(kept)} probes")
    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1)
    outs = llm.generate(kept, sp)

    def first_tok_lp(o):
        lps = o.outputs[0].logprobs
        if not lps:
            return None
        step = lps[0]
        tid = o.outputs[0].token_ids[0]
        return step[tid].logprob if tid in step else None

    results = []
    for rec in meta:
        qi = rec["qi"]; g = gen[qi]
        inter = []
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                n_out = outs[s + 2 * k]      # neutral
                ch_out = outs[s + 2 * k + 1]  # challenge
                inter.append({
                    "cut": c,
                    "neutral": extract_boxed_head(n_out.outputs[0].text),
                    "challenge": extract_boxed_head(ch_out.outputs[0].text),
                    "neutral_lp": first_tok_lp(n_out),
                })
        results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                        "final_answer": None, "n_segs": rec.get("n_segs", 0),
                        "intermediate": inter})

    out_path = EXP_ROOT / "challenge" / f"{args.gen_tag}_chal.json"
    save_json(results, out_path)
    print(f"[chal] saved -> {out_path}")


if __name__ == "__main__":
    main()
