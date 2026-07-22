"""Generate reasoning chains with Qwen3 (thinking mode) via vLLM.

Produces, per question:
  - primary sample: full text + per-token logprobs (for single-pass UQ baselines)
  - k-1 extra samples: text only (for self-consistency)
Correctness verified deterministically (no judge model).
Everything cached to EXP_ROOT so downstream analysis is generation-free.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from acd.env import EXP_ROOT, model_path, save_json
from acd import data as dv


PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "End your response with the final answer in \\boxed{{}}.\n\n{question}"
)


def build_messages(question: str):
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--k", type=int, default=8, help="samples per question (>=1)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1234, help="primary-sample decode seed")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    rows = dv.load_dataset(args.dataset, limit=args.limit)
    print(f"[gen] {args.dataset}: {len(rows)} questions, k={args.k}, model={args.model}")

    def render(msgs):
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(args.thinking),
            )
        except TypeError:
            # tokenizer without an enable_thinking kwarg (non-thinking models)
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    prompts = [render(build_messages(r["question"])) for r in rows]

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=max(4096, args.max_tokens + 2048),
              enforce_eager=False)

    # Primary sample: capture logprobs of chosen token (+alternatives for entropy proxy)
    sp_primary = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                                max_tokens=args.max_tokens, logprobs=5, seed=args.seed)
    out_primary = llm.generate(prompts, sp_primary)

    extra = None
    if args.k > 1:
        # capture logprobs on extra samples too -> per-sample confidence (CISC/DeepConf)
        sp_extra = SamplingParams(n=args.k - 1, temperature=args.temperature, top_p=args.top_p,
                                  max_tokens=args.max_tokens, logprobs=1, seed=args.seed + 100)
        extra = llm.generate(prompts, sp_extra)

    def _sample_conf(o):
        """mean and min chosen-token logprob for one sample output."""
        if o.logprobs is None:
            return None, None
        lps = []
        for step, tid in zip(o.logprobs, o.token_ids):
            if step and tid in step:
                lps.append(step[tid].logprob)
        if not lps:
            return None, None
        return float(sum(lps) / len(lps)), float(min(lps))

    results = []
    for i, r in enumerate(rows):
        po = out_primary[i].outputs[0]
        # per-token chosen logprob + top-k logprobs for entropy
        chosen_lp = []
        topk_lp = []
        if po.logprobs is not None:
            for step, tid in zip(po.logprobs, po.token_ids):
                if step is None:
                    continue
                # chosen token logprob
                if tid in step:
                    chosen_lp.append(step[tid].logprob)
                vals = sorted([v.logprob for v in step.values()], reverse=True)
                topk_lp.append(vals)
        samples = [po.text]
        # per-sample mean/min logprob: primary first, then extras
        pm, pmin = _sample_conf(po)
        sample_meanlp = [pm]
        sample_minlp = [pmin]
        if extra is not None:
            for o in extra[i].outputs:
                samples.append(o.text)
                m, mn = _sample_conf(o)
                sample_meanlp.append(m)
                sample_minlp.append(mn)
        results.append({
            "id": r["id"], "question": r["question"], "gold": r["gold"],
            "gold_raw": r.get("gold_raw", r["gold"]),
            "primary_text": po.text,
            "chosen_logprobs": chosen_lp,
            "topk_logprobs": topk_lp,
            "samples": samples,
            "sample_meanlp": sample_meanlp,
            "sample_minlp": sample_minlp,
            "n_gen_tokens": len(po.token_ids),
        })

    tag = args.tag or f"{args.dataset}_{args.model}_k{args.k}_n{len(rows)}"
    out_path = EXP_ROOT / "gen" / f"{tag}.json"
    save_json(results, out_path)
    print(f"[gen] saved {len(results)} -> {out_path}")


if __name__ == "__main__":
    main()
