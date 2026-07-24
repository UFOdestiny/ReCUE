"""Multi-cue confidence probe (P1-4 robustness + performance lever).

Same per-cut forced-answer protocol as run_probe_confidence, but elicits the answer
with THREE semantically-equivalent cues and records each cut's answer + first-token
logprob under every cue:
  cue0: "The final answer is \\boxed{"          (the original)
  cue1: "Therefore, the answer is \\boxed{"
  cue2: "So the final result is \\boxed{"

Output {cut, cues:[{cue_id, ans, lp}...]} -> lets downstream (i) test cue robustness
(mean/worst over cues) and (ii) test a cue-ENSEMBLE feature set as a performance lever.
Writes to conf_mc/<tag>_confmc.json. Idempotent. KV-cache reused across cues (shared
prefix) so marginal cost stays small.
"""
from __future__ import annotations

import argparse, json
from recue.env import EXP_ROOT, model_path, save_json
from recue.probe import split_think, segment_steps, cut_points, extract_boxed_head
from recue.generate import build_messages

CUES = ["The final answer is \\boxed{",
        "Therefore, the answer is \\boxed{",
        "So the final result is \\boxed{"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--max-probes", type=int, default=8); ap.add_argument("--probe-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--max-model-len", type=int, default=14336)
    args = ap.parse_args()

    out = EXP_ROOT / "conf_mc" / f"{args.gen_tag}_confmc.json"
    if out.exists(): print(f"[confmc] exists {out}"); return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    mpath = model_path(args.model); tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    flat, meta = [], []
    for qi, g in enumerate(gen):
        try:
            base = tok.apply_chat_template(build_messages(g["question"], mc=(g.get("type")=="mc")),
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            base = tok.apply_chat_template(build_messages(g["question"], mc=(g.get("type")=="mc")),
                                           tokenize=False, add_generation_prompt=True)
        ptext = g["primary_text"]
        if "<think>" in ptext: ot, ct = "<think>", "</think>"
        elif "[THINK]" in ptext: ot, ct = "[THINK]", "[/THINK]"
        else: ot, ct = None, ""
        is_think = ot is not None
        topen = "" if (ot and base.rstrip().endswith(ot)) else (ot + "\n" if is_think else "")
        segs = segment_steps(split_think(ptext))
        if not segs: meta.append({"qi": qi, "cuts": []}); continue
        cuts = cut_points(len(segs), args.max_probes)
        rec = {"qi": qi, "cuts": cuts, "start": len(flat)}
        for c in cuts:
            body = topen + "\n\n".join(segs[:c])
            for cue in CUES:
                tail = (f"\n{ct}\n\n{cue}" if is_think else f"\n\n{cue}")
                flat.append(base + body + tail)
        meta.append(rec)

    max_in = args.max_model_len - args.probe_tokens - 8
    kept = []
    for p in flat:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in: ids = ids[-max_in:]; p = tok.decode(ids)
        kept.append(p)
    print(f"[confmc] {len(gen)} q -> {len(kept)} probes ({len(CUES)} cues)")

    llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, enable_prefix_caching=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1)
    outs = llm.generate(kept, sp)

    def lp0(o):
        lps = o.outputs[0].logprobs
        if not lps: return None
        tid = o.outputs[0].token_ids[0]
        return lps[0][tid].logprob if tid in lps[0] else None

    nC = len(CUES); results = []
    for rec in meta:
        g = gen[rec["qi"]]; inter = []
        if rec.get("cuts"):
            s = rec["start"]
            for k, c in enumerate(rec["cuts"]):
                cues_out = []
                for ci in range(nC):
                    o = outs[s + k*nC + ci]
                    cues_out.append({"cue_id": ci, "ans": extract_boxed_head(o.outputs[0].text), "lp": lp0(o)})
                inter.append({"cut": c, "cues": cues_out})
        results.append({"id": g["id"], "gold": g["gold"], "gold_raw": g.get("gold_raw"), "intermediate": inter})
    save_json(results, out)
    print(f"[confmc] saved -> {out}")


if __name__ == "__main__":
    main()
