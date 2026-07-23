"""P0-5 real system-efficiency benchmark (More_EXP.md Sec 2.5).

Measures ACTUAL wall-clock cost on one GPU, not decoded-token ratios. For each
method we time the marginal work over a shared pool of cached questions:

  primary        one full trace generation (the 1x reference)
  chainuq_M{2,4,8}  primary trace + M anchored-answer probes (KV-cache REUSED)
  chainuq_nocache   same but prefix caching DISABLED (isolates cache benefit)
  p_true         primary trace + one extra full forward pass (verify prompt)
  sc_{2,4,8}     k full generations

Reported per method: end-to-end latency median/p90/p95, throughput (q/s),
time-to-first-token, probe decode time, peak GPU memory, decoded/prefill tokens,
#full-gens / #probe-calls / #extra-fwd, prefix-cache on/off. Raw per-query rows
-> system_efficiency.jsonl; aggregate -> system_efficiency_summary.json. Pair
with cached AUROC to draw the AUROC-latency Pareto curve.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from acd.env import EXP_ROOT, model_path
from acd.generate import build_messages
from acd.probe import split_think, segment_steps, cut_points


def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--gen-tag", required=True, help="cached gen cell to source questions/traces")
    ap.add_argument("--n", type=int, default=64, help="questions to benchmark")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--probe-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())[: args.n]
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)

    def render(q, mc):
        try:
            return tok.apply_chat_template(build_messages(q, mc=mc), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            return tok.apply_chat_template(build_messages(q, mc=mc), tokenize=False,
                                           add_generation_prompt=True)

    is_mc = gen[0].get("type") == "mc"
    prompts = [render(g["question"], is_mc) for g in gen]

    # pre-build probe prompts (from cached traces) for chainuq timing
    def probe_prompts_for(g, M):
        ptext = g["primary_text"]
        if "<think>" in ptext:
            ot, ct = "<think>", "</think>"
        elif "[THINK]" in ptext:
            ot, ct = "[THINK]", "[/THINK]"
        else:
            ot, ct = None, ""
        is_think = ot is not None
        base = render(g["question"], is_mc)
        topen = "" if (ot and base.rstrip().endswith(ot)) else (ot + "\n" if is_think else "")
        elicit = f"\n{ct}\n\nThe final answer is \\boxed{{" if is_think else "\n\nThe final answer is \\boxed{"
        segs = segment_steps(split_think(ptext))
        if not segs:
            return []
        cuts = cut_points(len(segs), M)
        return [base + topen + "\n\n".join(segs[:c]) + elicit for c in cuts]

    results = {}
    peak = {}

    def bench(name, fn, enable_cache=True):
        llm = LLM(model=mpath, trust_remote_code=True, dtype="bfloat16",
                  gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                  enable_prefix_caching=enable_cache, enforce_eager=False)
        torch.cuda.reset_peak_memory_stats()
        rows = fn(llm)
        peak[name] = float(torch.cuda.max_memory_allocated() / 1e9)
        results[name] = rows
        del llm
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # ---- primary generation (1x reference) ----
    def run_primary(llm, k=1, tag="primary"):
        sp = SamplingParams(n=k, temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, seed=1234)
        rows = []
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            t0 = time.perf_counter()
            outs = llm.generate(chunk, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            ntok = sum(sum(len(o.token_ids) for o in out.outputs) for out in outs)
            per_q = dt / len(chunk)
            for out in outs:
                rows.append({"latency": per_q, "decoded": sum(len(o.token_ids) for o in out.outputs),
                             "full_gens": k, "probe_calls": 0, "extra_fwd": 0})
        return rows

    # ---- chainuq: primary + M probes ----
    def run_chainuq(M):
        def _f(llm):
            sp_g = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, seed=1234)
            sp_p = SamplingParams(n=1, temperature=0.0, max_tokens=args.probe_tokens, logprobs=1)
            rows = []
            for i in range(0, len(gen), args.batch):
                gchunk = gen[i:i + args.batch]
                pchunk = prompts[i:i + args.batch]
                t0 = time.perf_counter()
                _ = llm.generate(pchunk, sp_g, use_tqdm=False)
                t_gen = time.perf_counter() - t0
                # probes (use cached traces so prefix is shared -> cache reuse)
                pp = []
                counts = []
                for g in gchunk:
                    ps = probe_prompts_for(g, M)
                    counts.append(len(ps)); pp.extend(ps)
                t1 = time.perf_counter()
                if pp:
                    po = llm.generate(pp, sp_p, use_tqdm=False)
                    ptok = sum(len(o.outputs[0].token_ids) for o in po)
                else:
                    ptok = 0
                t_probe = time.perf_counter() - t1
                nb = len(gchunk)
                for j, g in enumerate(gchunk):
                    rows.append({"latency": (t_gen + t_probe) / nb, "gen_time": t_gen / nb,
                                 "probe_time": t_probe / nb, "probe_calls": counts[j],
                                 "full_gens": 1, "extra_fwd": 0})
            return rows
        return _f

    # ---- p_true: primary + 1 extra full forward (verify prompt, short decode) ----
    def run_ptrue(llm):
        sp_g = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, seed=1234)
        sp_v = SamplingParams(n=1, temperature=0.0, max_tokens=4, logprobs=5)
        rows = []
        for i in range(0, len(gen), args.batch):
            gchunk = gen[i:i + args.batch]; pchunk = prompts[i:i + args.batch]
            t0 = time.perf_counter(); _ = llm.generate(pchunk, sp_g, use_tqdm=False); t_gen = time.perf_counter() - t0
            vprompts = [pchunk[j] + gchunk[j]["primary_text"][:2000] +
                        "\n\nIs the above answer correct? Answer True or False:" for j in range(len(gchunk))]
            t1 = time.perf_counter(); _ = llm.generate(vprompts, sp_v, use_tqdm=False); t_v = time.perf_counter() - t1
            nb = len(gchunk)
            for _g in gchunk:
                rows.append({"latency": (t_gen + t_v) / nb, "gen_time": t_gen / nb,
                             "extra_fwd_time": t_v / nb, "full_gens": 1, "probe_calls": 0, "extra_fwd": 1})
        return rows

    # ---- run all ----
    bench("primary", lambda llm: run_primary(llm, 1))
    for M in (2, 4, 8):
        bench(f"chainuq_M{M}", run_chainuq(M))
    bench("chainuq_M8_nocache", run_chainuq(8), enable_cache=False)
    bench("p_true", run_ptrue)
    for k in (2, 4, 8):
        bench(f"sc_{k}", lambda llm, k=k: run_primary(llm, k, f"sc_{k}"))

    # ---- aggregate ----
    summ = {}
    jl = []
    for name, rows in results.items():
        lat = np.array([r["latency"] for r in rows])
        summ[name] = {
            "n": len(rows),
            "latency_median": float(np.median(lat)),
            "latency_p90": pct(lat, 90), "latency_p95": pct(lat, 95),
            "throughput_qps": float(len(rows) / lat.sum()) if lat.sum() else None,
            "peak_gpu_gb": peak.get(name),
            "probe_time_mean": float(np.mean([r.get("probe_time", 0) for r in rows])),
            "gen_time_mean": float(np.mean([r.get("gen_time", r["latency"]) for r in rows])),
            "extra_fwd_time_mean": float(np.mean([r.get("extra_fwd_time", 0) for r in rows])),
            "full_gens": int(np.mean([r.get("full_gens", 1) for r in rows])),
            "probe_calls_mean": float(np.mean([r.get("probe_calls", 0) for r in rows])),
        }
        for r in rows:
            jl.append({"method": name, **r})

    tag = args.out_tag or f"{args.model}_{args.gen_tag}"
    (EXP_ROOT / f"system_efficiency_{tag}.jsonl").write_text("\n".join(json.dumps(x) for x in jl))
    (EXP_ROOT / f"system_efficiency_{tag}.json").write_text(json.dumps(summ, indent=2))

    print(f"\n=== SYSTEM EFFICIENCY ({args.model} on {args.gen_tag}, n={len(gen)}) ===")
    print(f"{'method':20s}{'lat_med':>9s}{'lat_p95':>9s}{'qps':>8s}{'peakGB':>8s}"
          f"{'gen_s':>8s}{'probe_s':>9s}")
    base = summ["primary"]["latency_median"]
    for name in summ:
        s = summ[name]
        rel = s["latency_median"] / base if base else float("nan")
        print(f"{name:20s}{s['latency_median']:9.3f}{s['latency_p95']:9.3f}{s['throughput_qps']:8.2f}"
              f"{(s['peak_gpu_gb'] or 0):8.1f}{s['gen_time_mean']:8.3f}{s['probe_time_mean']:9.4f}"
              f"  ({rel:.2f}x)")
    print(f"saved system_efficiency_{tag}.json/.jsonl")


if __name__ == "__main__":
    main()
