"""P(True) self-evaluation baseline (Kadavath et al. 2022), judge-free.

The SAME model self-scores its own answer with a single extra short forward pass.
This is the reviewer-requested 'self-verdict' baseline. No external judge.

For each cached primary generation, build a prompt:
  <question> ... Proposed answer: <final_answer>.
  Is the proposed answer correct?
and read the probability mass on 'True' vs 'False' at the next token.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

from recue.env import EXP_ROOT, model_path, save_json, extract_pred_math
from recue.generate import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=12288)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    prompts = []
    for g in gen:
        ans = extract_pred_math(g["primary_text"]) or "(no answer)"
        msgs = build_messages(g["question"])
        # append the model's own answer + a verification question as a follow-up user turn
        msgs = msgs + [
            {"role": "assistant", "content": f"The final answer is \\boxed{{{ans}}}."},
            {"role": "user", "content": (
                "Is the proposed final answer correct? "
                "Respond with a single word: True or False.")},
        ]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # truncate overlong
    max_in = args.max_model_len - 8
    kept = []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            ids = ids[-max_in:]
            p = tok.decode(ids)
        kept.append(p)

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate(kept, sp)

    # find P(True) vs P(False) from first-token top-k logprobs
    def true_prob(o):
        step = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        pt, pf = 0.0, 0.0
        for tid, lp in step.items():
            tokstr = (lp.decoded_token or "").strip().lower()
            if tokstr in ("true", "yes", "correct"):
                pt += np.exp(lp.logprob)
            elif tokstr in ("false", "no", "incorrect", "wrong"):
                pf += np.exp(lp.logprob)
        if pt + pf < 1e-9:
            return 0.5
        return float(pt / (pt + pf))

    scores = {g["id"]: true_prob(o) for g, o in zip(gen, outs)}
    out_path = EXP_ROOT / "ptrue" / f"{args.gen_tag}_ptrue.json"
    save_json(scores, out_path)
    print(f"[ptrue] saved {len(scores)} -> {out_path}")


if __name__ == "__main__":
    main()
