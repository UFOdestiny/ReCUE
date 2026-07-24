"""Rebuttal probes (cache-extending) for two reviewer concerns.

Both probe ONLY at the FULL reasoning prefix (all segments kept), reusing the
same base+think+elicit construction as scripts/run_probe_confidence.py, so they
are directly comparable to the existing ReCUE probes and stay judge-free.

Concern 3.2b  -- "multi-prefix vs. just multiple observations".
  REPEATED-ENDPOINT control: at the SINGLE full prefix, draw R stochastic
  short-answer decodes (temperature>0). This is observation-count-matched to
  ReCUE's M prefix probes but with ZERO reasoning-depth variation. Records each
  repeat's boxed answer + first-answer-token logprob so the analysis can build the
  exact same identity/confidence commitment features from repeated endpoint probes.

Concern 3.4  -- "first-answer-token confidence is fragile".
  FULL-ANSWER LIKELIHOOD: at the full prefix, one greedy decode of the whole boxed
  answer with per-token logprobs. Records the answer, its token count, and the
  cumulative logprobs so the analysis can form length-normalized complete-answer
  log-likelihood and the joint likelihood of the first 2/3 tokens.

Output schema (EXP_ROOT/rebuttal/{tag}_reb.json), one record per question:
  {id, gold, gold_raw, n_segs,
   repeat: [{answer, lp0}, ... R],            # stochastic endpoint repeats
   full:   {answer, ntok, tok_lps:[...]}}     # greedy full-answer decode w/ logprobs
"""
from __future__ import annotations

import argparse
import json

from recue.env import EXP_ROOT, model_path, save_json
from recue.probe import split_think, segment_steps, extract_boxed_head
from recue.generate import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--repeat-tokens", type=int, default=16)
    ap.add_argument("--full-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = EXP_ROOT / "rebuttal" / f"{args.gen_tag}_reb.json"
    if out.exists():
        print(f"[reb] exists {out}"); return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    # Build ONE full-prefix prompt per question (identical construction to conf probe).
    prompts, meta = [], []
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
            meta.append({"qi": qi, "ok": False}); continue
        prompt = base + topen + "\n\n".join(segs) + elicit  # FULL prefix (all segments)
        meta.append({"qi": qi, "ok": True, "idx": len(prompts)})
        prompts.append(prompt)

    # Truncate overflow prompts (keep the elicit tail) exactly like the conf probe.
    max_in = args.max_model_len - max(args.repeat_tokens, args.full_tokens) - 8
    kept = []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in:
            ids = ids[-max_in:]; p = tok.decode(ids)
        kept.append(p)
    prompts = kept
    print(f"[reb] {len(gen)} q -> {len(prompts)} full-prefix prompts "
          f"(x{args.repeats} stochastic + 1 greedy)")

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)

    def lp0(o):
        lps = o.outputs[0].logprobs
        if not lps:
            return None
        tid = o.outputs[0].token_ids[0]
        return lps[0][tid].logprob if tid in lps[0] else None

    # (a) R stochastic short-answer repeats at the same full prefix (n=repeats).
    sp_rep = SamplingParams(n=args.repeats, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.repeat_tokens, logprobs=1, seed=args.seed)
    reps = llm.generate(prompts, sp_rep)

    # (b) One greedy full-answer decode with per-token logprobs.
    sp_full = SamplingParams(n=1, temperature=0.0, max_tokens=args.full_tokens, logprobs=1)
    fulls = llm.generate(prompts, sp_full)

    def rep_records(o):
        recs = []
        for comp in o.outputs:
            ans = extract_boxed_head(comp.text)
            l0 = None
            if comp.logprobs:
                tid = comp.token_ids[0]
                if tid in comp.logprobs[0]:
                    l0 = comp.logprobs[0][tid].logprob
            recs.append({"answer": ans, "lp0": l0})
        return recs

    def full_record(o):
        comp = o.outputs[0]
        ans = extract_boxed_head(comp.text)
        # per-token chosen logprobs over the decoded answer (until the closing brace
        # region -- we keep the full window; analysis truncates at the answer length).
        tok_lps = []
        if comp.logprobs:
            for step, tid in zip(comp.logprobs, comp.token_ids):
                if tid in step:
                    tok_lps.append(step[tid].logprob)
        return {"answer": ans, "ntok": len(comp.token_ids), "tok_lps": tok_lps}

    results = []
    for rec in meta:
        g = gen[rec["qi"]]
        if not rec["ok"]:
            results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                            "n_segs": 0, "repeat": [], "full": None})
            continue
        i = rec["idx"]
        results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"),
                        "n_segs": len(segment_steps(split_think(g["primary_text"]))),
                        "repeat": rep_records(reps[i]), "full": full_record(fulls[i])})
    save_json(results, out)
    print(f"[reb] saved -> {out}")


if __name__ == "__main__":
    main()
