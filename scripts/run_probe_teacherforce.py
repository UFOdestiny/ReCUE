"""Teacher-forced original-answer support probe (reviewer concern 3).

At the SAME completed reasoning prefix used by ARC, teacher-force the ORIGINAL
returned answer a and record the model's support for it, without decoding a new
answer. This isolates "re-generate another answer and compare identity" (ARC)
from "how strongly does the model support the answer it already gave" (support).

For each question we build base+think+segments+elicit exactly as the ARC/rebuttal
probe, append the original answer string a (from sampans modal answer) plus the
closing brace, and read vLLM prompt_logprobs over the appended answer tokens.

Records (EXP_ROOT/tforce/{tag}_tf.json), one record per question:
  {id, ans, ntok, tok_lps:[...], first_lp}
where tok_lps are per-token logprobs of the teacher-forced answer tokens.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from recue.env import EXP_ROOT, model_path
from recue.probe import split_think, segment_steps
from recue.generate import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    out = EXP_ROOT / "tforce" / f"{args.gen_tag}_tf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[tf] exists {out}"); return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    saf = EXP_ROOT / "sampans" / f"{args.gen_tag}.json"
    sa = json.loads(saf.read_text()) if saf.exists() else {}
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    prompts, meta = [], []
    for qi, g in enumerate(gen):
        rid = g["id"]
        ans_list = [a for a in sa.get(rid, []) if a is not None]
        if not ans_list:
            meta.append({"qi": qi, "ok": False}); continue
        a = Counter(ans_list).most_common(1)[0][0]   # modal original answer
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
        prefix = base + topen + "\n\n".join(segs) + elicit
        # count how many tokens the answer string occupies when appended
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        full_ids = tok(prefix + a + "}", add_special_tokens=False)["input_ids"]
        n_ans = len(full_ids) - len(pre_ids)
        # truncate overflow from the LEFT keeping prefix tail + answer
        max_in = args.max_model_len - 8
        if len(full_ids) > max_in:
            full_ids = full_ids[-max_in:]
            n_ans = min(n_ans, len(full_ids) - 1)
        meta.append({"qi": qi, "ok": True, "idx": len(prompts), "n_ans": max(1, n_ans), "ans": a})
        prompts.append(tok.decode(full_ids))

    print(f"[tf] {len(gen)} q -> {len(prompts)} teacher-forced prompts")
    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    # prompt_logprobs=0 -> logprob of each actual prompt token
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate(prompts, sp)

    recs = {}
    for m in meta:
        if not m["ok"]:
            continue
        o = outs[m["idx"]]
        pl = o.prompt_logprobs  # list aligned to prompt tokens; entry 0 is None
        n_ans = m["n_ans"]
        tids = o.prompt_token_ids
        tok_lps = []
        # last n_ans+1 tokens correspond to answer + closing brace; take answer tokens
        span = range(len(tids) - n_ans - 1, len(tids) - 1)
        for pos in span:
            if pos <= 0 or pl[pos] is None:
                continue
            tid = tids[pos]
            entry = pl[pos]
            if tid in entry:
                tok_lps.append(entry[tid].logprob)
        if not tok_lps:
            continue
        recs[m["qi"]] = {"ans": m["ans"], "ntok": len(tok_lps),
                         "tok_lps": tok_lps, "first_lp": tok_lps[0]}

    # write keyed by question id
    byid = {}
    for m in meta:
        if m["ok"] and m["qi"] in recs:
            byid[gen[m["qi"]]["id"]] = recs[m["qi"]]
    out.write_text(json.dumps(byid))
    print(f"[tf] wrote {len(byid)} records -> {out}")


if __name__ == "__main__":
    main()
