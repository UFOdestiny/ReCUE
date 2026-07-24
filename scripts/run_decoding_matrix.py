"""Exp1: decoding-symmetry matrix (reviewer concern 1).

For one (model, dataset) cell, regenerate the PRIMARY reasoning trace under a chosen
temperature, then run the ARC re-elicitation probe under a chosen probe temperature,
verify both answers judge-free, and dump per-question records so the analysis can
build ARC/ReCUE features and answer-transition rates.

Conditions are given as (primary_temp, probe_temp) pairs. The decisive one is
(0.0, 0.0): greedy primary + greedy probe. If ARC still ranks correctness there,
agreement is not merely sampling noise.

Output EXP_ROOT/decmatrix/{tag}__p{pt}_q{qt}.json:
  {id, gold, primary_ans, primary_correct, reelicit_ans, reelicit_correct,
   agree, first_lp, ans_lps:[...], chosen_lp:[...], topk_lp:[...]}
"""
from __future__ import annotations
import argparse, json
import numpy as np
from recue.env import EXP_ROOT, model_path, verify_math, extract_pred_math
from recue.probe import split_think, segment_steps, extract_boxed_head
from recue.generate import build_messages
from recue import data as dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--primary-temp", type=float, required=True)
    ap.add_argument("--probe-temp", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--probe-tokens", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    pt = f"{args.primary_temp:g}".replace(".", "")
    qt = f"{args.probe_temp:g}".replace(".", "")
    out = EXP_ROOT / "decmatrix" / f"{args.tag}__p{pt}_q{qt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[dm] exists {out}"); return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = dv.load_dataset(args.dataset, limit=args.limit)
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [render(build_messages(r["question"], mc=(r.get("type") == "mc"))) for r in rows]
    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)

    # ---- primary generation ----
    greedy_p = args.primary_temp == 0.0
    sp_p = SamplingParams(n=1, temperature=args.primary_temp, top_p=(1.0 if greedy_p else 0.95),
                          max_tokens=args.max_tokens, logprobs=5, seed=args.seed)
    outs_p = llm.generate(prompts, sp_p)

    # ---- build ARC probe prompts at completed prefix ----
    probe_prompts, meta = [], []
    for qi, (r, o) in enumerate(zip(rows, outs_p)):
        po = o.outputs[0]
        ptext = po.text
        primary_ans = extract_pred_math(ptext)
        # per-token chosen + topk logprobs for TUP
        chosen_lp, topk_lp = [], []
        if po.logprobs is not None:
            for step, tid in zip(po.logprobs, po.token_ids):
                if step is None:
                    continue
                if tid in step:
                    chosen_lp.append(step[tid].logprob)
                topk_lp.append(sorted([v.logprob for v in step.values()], reverse=True))
        base = render(build_messages(r["question"], mc=(r.get("type") == "mc")))
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
        rec = {"qi": qi, "gold": r.get("gold"), "primary_ans": primary_ans,
               "chosen_lp": chosen_lp, "topk_lp": topk_lp, "ok": bool(segs)}
        if not segs:
            meta.append(rec); continue
        prompt = base + topen + "\n\n".join(segs) + elicit
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        max_in = args.max_model_len - args.probe_tokens - 8
        if len(ids) > max_in:
            ids = ids[-max_in:]; prompt = tok.decode(ids)
        rec["idx"] = len(probe_prompts)
        meta.append(rec)
        probe_prompts.append(prompt)

    # ---- ARC probe ----
    greedy_q = args.probe_temp == 0.0
    sp_q = SamplingParams(n=1, temperature=args.probe_temp, top_p=(1.0 if greedy_q else 0.95),
                          max_tokens=args.probe_tokens, logprobs=1, seed=args.seed + 7)
    outs_q = llm.generate(probe_prompts, sp_q)

    records = []
    for rec in meta:
        if not rec["ok"]:
            continue
        o = outs_q[rec["idx"]]
        comp = o.outputs[0]
        re_ans = extract_boxed_head(comp.text)
        ans_lps = []
        if comp.logprobs:
            for step, tid in zip(comp.logprobs, comp.token_ids):
                if tid in step:
                    ans_lps.append(step[tid].logprob)
        gold = rec["gold"]
        pc = verify_math(rec["primary_ans"], gold) if rec["primary_ans"] else 0
        rc = verify_math(re_ans, gold) if re_ans else 0
        agree = 1 if (rec["primary_ans"] and re_ans and verify_math(rec["primary_ans"], re_ans)) else 0
        records.append({
            "id": rows[rec["qi"]]["id"], "gold": gold,
            "primary_ans": rec["primary_ans"], "primary_correct": int(pc),
            "reelicit_ans": re_ans, "reelicit_correct": int(rc), "agree": agree,
            "first_lp": ans_lps[0] if ans_lps else None, "ans_lps": ans_lps,
            "chosen_lp": rec["chosen_lp"], "topk_lp": rec["topk_lp"],
        })
    out.write_text(json.dumps(records))
    acc = np.mean([r["primary_correct"] for r in records]) if records else 0
    agr = np.mean([r["agree"] for r in records]) if records else 0
    print(f"[dm] {args.tag} p{pt}_q{qt}: {len(records)} recs, acc={acc:.3f}, agree={agr:.3f} -> {out}")


if __name__ == "__main__":
    main()
