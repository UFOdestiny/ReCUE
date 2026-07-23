# ChainUQ — KDD-hardening experiments (results log)

New experiments run per `docs/More_EXP.md` to close the most likely rejection
paths. All are judge-free, reuse the frozen 31-cell cache (7 model families × 5
math datasets; k=8), and write source JSON under `$EXP_ROOT`. Non-math (P0-2) and
system efficiency (P0-5) add new generated cells. Numbers below are copied from the
saved JSON/logs named in each section. **Skipped** = already existed and was not
re-run.

Reproduce: scripts live in `experiments/`; run via the commands in each section.

---

## P0-3 — Temporal-ORDER ablation (decisive novelty test)
`experiments/ablation_temporal_order.py` → `ablation_order.json` / `.log`
(31 cells, 5-seed CV, 10 permutation seeds, 1000-sample paired bootstrap)

Holds the marginal information (confidence multiset + identity counts) fixed and
destroys ONLY order.

| config | macro AUROC | meaning |
|---|:--:|---|
| FINAL | 0.629 | final-probe confidence only |
| CONV | 0.645 | answer-identity convergence (prior-art content) |
| C+F | 0.695 | CONV + FINAL (endpoint null) |
| BAG | 0.701 | order-invariant confidence+identity multiset |
| PERM | 0.741 | ordered features on randomly permuted seqs (train-perm→test-perm, 10 seeds) |
| REVERSE | 0.754 | time-reversed |
| ORD-CONF | 0.730 | confidence trajectory only |
| ORD-ID | 0.645 | identity trajectory only |
| **DUAL** | **0.743** | ordered dual trajectory |
| **FULL** | **0.785** | DUAL + sequence features |

Primary contrasts (mean Δ, #significant/31):
- **DUAL − BAG = +0.042 (11/31 sig)** — ordered features beat the unordered multiset.
- **DUAL − PERM = +0.003 (3/31 sig)** — destroying order barely hurts once the
  model is retrained on permuted data.
- **FULL − C+F = +0.090 (18/31 sig)** — full method dominates the endpoint null.

**Verdict (honest, per More_EXP §2.3 Go/No-go):** the gain is genuinely from the
*multi-prefix commitment statistics*, not strictly from *temporal order*. Reposition
the claim from "temporal dynamics" to **"multi-prefix answer-commitment evidence"**;
keep the ordered features (they beat BAG) but do not over-claim order-sensitivity.

---

## P0-1 — Transfer: does the signal transfer? (`experiments/transfer.py`)
Head trained on SOURCE cells, evaluated on unseen TARGET cells; normalization,
feature selection, and classifier fit on source only; no dataset-id / accuracy /
target statistics enter the head. Baselines transferred on the same split.
Outputs `transfer_lodo.json`, `transfer_lomo.json`, `transfer_global.json`.

### P0-1A — Leave-one-DATASET-out (LODO) — **STRONG**
- Macro AUROC **0.796**, worst-domain 0.550.
- Mean degradation vs in-domain per-cell head = **−0.004** (essentially none).
- ChainUQ > best transferred baseline on **22/30** held-out domains, **10/30
  significant** (paired bootstrap). Biggest wins on olympiad (+0.06–0.08) and
  math500 (+0.07) across Qwen/Phi families.
- Directly refutes "per-cell supervised feature engineering."

### P0-1B — Leave-one-model-FAMILY-out (LOMO) — **MIXED (as anticipated)**
| held-out family | ChainUQ AUROC | Δ vs base | p |
|---|:--:|:--:|:--:|
| Qwen3 | 0.761 | **+0.025** | **0.000** |
| Phi4 | 0.740 | +0.001 | 0.472 |
| Ministral | 0.675 | −0.065 | 1.000 |
| Qwen3.5 | 0.785 | −0.068 | 1.000 |
| **size** Qwen3-4B/8B → 14B | **0.836** | **+0.042** | **0.000** |

Intra-family (Qwen3) and size transfer are positive & significant; transfer *to*
Ministral / Qwen3.5 is negative. **Claim cross-dataset transfer, not universal
cross-model calibration** (More_EXP §2.2 "acceptable" band).

### P0-1C — Global unified head — **STRONG**
- Leave-one-cell-out global head (**no cell identity**): macro **0.767**, worst 0.529.
- Global + cell-id (upper bound): macro 0.765 → **cell identity adds nothing**; one
  shared head recovers ~97% of the per-cell macro (0.791). Single ChainUQ head works.

---

## P0-6 — Statistics upgrade (`experiments/stats_upgrade.py` → `stats_upgrade.json`)
Pre-registered primary contrasts; per-cell paired bootstrap + Holm over 31 cells +
cell-equal **hierarchical bootstrap** (2000 resamples).

| contrast | mean Δ | sig (uncorr / Holm) | hierarchical Δ [95% CI] p |
|---|:--:|:--:|:--:|
| C1 ChainUQ vs best 1x | +0.032 | 18/31 / 8/31 | +0.033 [−0.018,+0.084] p=0.103 |
| **C2 ordered-traj vs CONV+FINAL** | +0.023 | 6/31 / 1/31 | **+0.023 [+0.007,+0.045] p=0.0015** |
| **C3 ChainUQ⊕SC@8 vs SC@8** | +0.016 | 11/31 / 3/31 | **+0.016 [+0.002,+0.031] p=0.013** |

AUROC summaries (macro / macro-no-AMC23 / worst-dataset):
- best-1x 0.759 / 0.765 / 0.510 · C+F 0.696 / 0.688 / 0.278 · ordered-traj 0.719 /
  0.704 / 0.444 · **ChainUQ 0.791 / 0.783 / 0.458** · SC@8 0.882 / 0.866 / 0.702 ·
  **fusion 0.898 / 0.888 / 0.736**.

**Read:** C2 and C3 are robustly significant under the strictest (cell-equal) test.
C1 dominates most cells but the pooled cell-equal effect is not significant — keep
the honest "best same-cost estimator, matches SC@2–4" framing, do not claim a
universal 1x win. Removing AMC23 barely changes macro (0.791→0.783).

---

## P1-1 — Self-consistency blind-spot stress test (`experiments/novelty_amplifiers.py --exp stress` → `amp_stress.json`)
Pre-registered vote-fraction thresholds. On high-consensus subsets SC-AUROC → 0.5
by construction; ChainUQ still ranks correct-vs-wrong.

| vote ≥ | cells | total n | wrong | SC AUROC | ChainUQ | fusion |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.625 | 26 | 9874 | 948 | 0.788 | 0.738 | 0.818 |
| 0.750 | 26 | 9145 | 591 | 0.737 | 0.703 | 0.770 |
| 0.875 | 24 | 7816 | 340 | 0.594 | 0.660 | 0.664 |
| **1.000** | 24 | 7138 | **244** | **0.500** | **0.637** | 0.590 |

At **unanimous** consensus (SC blind), ChainUQ still scores 0.637 macro AUROC and
ranks the 244 confident-consensus errors (≥0.65 on 11/24 cells). This is the
mechanistic reason matched-budget fusion (C3) beats SC@8. **Promote to main text.**

---

## P1-3 — Label efficiency (`--exp labeleff` → `amp_labeleff.json`)
Train on {1,2,5,10,25,50,100}% of a cell's labeled problems, fixed 40% test fold.

| label frac | ChainUQ | seq-only | CONV+FINAL |
|---|:--:|:--:|:--:|
| 1% | 0.736 | 0.651 | 0.691 |
| 5% | 0.741 | 0.652 | 0.694 |
| 25% | 0.763 | 0.668 | 0.711 |
| 100% | 0.796 | 0.674 | 0.737 |

ChainUQ reaches **95% of full-data AUROC at ~25% labels**; at just **1% labels
(0.736) it already exceeds CONV+FINAL at 100% (0.737)**. Verifier-label demand is low.

---

## P0-4 — Classifier-capacity control (`--exp capacity` → `amp_capacity.json`)
Same ChainUQ features, different heads (macro AUROC): logistic **0.789**, RF 0.817,
GBT 0.799, MLP 0.752. Heads are close (RF marginally best), so the gain comes from
the **observation (features)**, not classifier capacity. Keep logistic in the paper
for simplicity; report RF as a capacity check.

---

## PERF — Performance levers toward SOTA (`experiments/perf_levers.py` → `perf_levers.json`)
Honest 5-seed 5-fold OOF over 31 math cells. Goal: how far can ChainUQ be pushed?

**1× tier (base ChainUQ = 0.791 macro AUROC):**
| lever | macro | Δ base | wins |
|---|:-:|:-:|:-:|
| base (logistic) | 0.791 | — | — |
| **base + P(True) feature** | **0.821** | **+0.038** | 20/25 |
| **base + RandomForest head** | **0.826** | **+0.035** | 23/31 |
| base + rich trajectory feats | 0.785 | −0.006 | 13/31 |
| base + GBT head | 0.697 | −0.094 | — |

**Sampling tier (SC@8 = 0.882 = the SOTA reference):**
| lever | macro | Δ SC@8 | wins |
|---|:-:|:-:|:-:|
| **fusion (ChainUQ ⊕ SC vote+entropy, logistic)** | **0.898** | **+0.016** | 24/31 |
| fusion + GBT | 0.760 | −0.122 | — |

**Read (honest):**
- **1× can be lifted 0.791 → 0.821 via P(True)-fusion** (extra-forward cost tier; two
  complementary signals; simple logistic). Still below SC@8 (0.882) — a single
  generation structurally cannot match 8× sampling. RF gives a similar +0.035 but is a
  capacity lever (verify it survives LODO before claiming; risk of in-domain overfit).
- **matched-8×-budget fusion 0.898 > SC@8 0.882 (+0.016, wins 24/31, worst −0.052)** —
  this is the defensible **"beats SOTA at matched budget"** result. Report here, not at 1×.
- **GBT/HistGBT numbers are unreliable** (piecewise-constant probs + 25-fold averaging
  breaks AUROC ranking) — do NOT use tree-boosting scores as-is; RF is fine (bagged →
  smoother). Keep logistic as the paper's default (simple, matches "not a patchwork").

## PERF-2 — Cost-tiered main table + lever verification (the SOTA question, settled)

**Cost-tiered table** (`experiments/main_table_v2.py` → `main_table_v2.json`), macro
AUROC, methods placed in FAIR cost blocks + pre-registered cell-equal hierarchical
bootstrap:

| tier | method | macro | contrast (hier. bootstrap) |
|---|---|:-:|---|
| 1× single-trace | mean_logprob / self_certainty | 0.682 / 0.684 | |
| 1× single-trace | **ChainUQ** | **0.791** | **T1 vs best-1× +0.103, CI[+0.056,+0.146], p=0.000 SIG** |
| +1 forward | p_true | 0.755 | |
| +1 forward | **ChainUQ+P(True)** | **0.821** | **T2 vs P(True) +0.066, CI[+0.008,+0.122], p=0.011 SIG** |
| 8× sampling | sc@8 | 0.882 | |
| 8× sampling | **fusion (ChainUQ⊕SC@8)** | **0.898** | **T3 vs SC@8 +0.016, CI[+0.002,+0.031], p=0.013 SIG** |

Placing P(True) in its own +1-forward tier (fair) makes ChainUQ the clear, significant
winner WITHIN the true 1× tier (+0.103) — cleaner than the earlier pooled C1 that
mixed P(True) into 1×.

**Lever verification under domain shift** (`experiments/transfer_head_capacity.py` →
`transfer_head_capacity.json`), LODO:
| lever | in-domain Δ | LODO transfer Δ | verdict |
|---|:-:|:-:|---|
| **+P(True) feature (logistic)** | +0.038 | **+0.042 (21/25)** | **robust — feature it** |
| RandomForest head | +0.035 | +0.015 (18/30) | capacity; ~60% is in-domain overfit, report cautiously |

**Cue-ensemble lever** (`experiments/cue_ensemble.py`, 6-cell multi-cue subset,
`conf_mc/` caches from `scripts/run_probe_multicue.py`): 3 semantically-equivalent
probe cues. cue_mean −0.015 vs best single cue, cue_ensemble +0.000 — **no performance
gain**, but a strong **robustness** result: cue spread only **0.018** (worst-cue 0.691
vs mean 0.698). ChainUQ is prompt-cue-invariant; report as robustness (P1-4), not a
lever.

**Bottom line on "can it reach SOTA":** 1× cannot reach SC@8 (0.882) — structural.
Best honest 1× = 0.821 (+P(True), significant, transfers). **Matched-8×-budget fusion
0.898 > SC@8 0.882 is the defensible beats-SOTA result** (significant, 24/31 cells).

## P0-2 — Non-math generalization (`experiments/non_math_generalization.py`)
BBH logical-deduction / tracking-shuffled-objects / date-understanding + GPQA-diamond
× {Qwen3-8B, Phi-4-reasoning, Ministral-3-14B-Reasoning}. Judge-free MC exact-match
(`acd/data_nonmath.py`). 12 new generated+probed cells. Outputs
`nonmath_indomain.json`, `nonmath_transfer.json`.

**Note on saturation:** BBH logical/tracking are near-solved by these strong
reasoning models (0–9 errors), so they cannot yield a meaningful AUROC and are
excluded from macro (logged, not silently dropped). The evaluation is carried by
**GPQA-diamond** (acc 0.51–0.58, 83–97 errors per cell — well balanced) and
**bbh_date** (14–31 errors). 5 cells clear the ≥15-error bar.

### Layer 1 — in-domain non-math (5-seed CV AUROC, cells with ≥15 errors)
| cell | n | acc | wrong | logp | self-cert | C+F | **ChainUQ** | SC@8 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| gpqa_qwen8b | 198 | .510 | 97 | .684 | .687 | .682 | **.768** | .796 |
| gpqa_phi4r | 198 | .581 | 83 | .589 | .617 | .556 | **.648** | .745 |
| gpqa_ministral | 197 | .533 | 92 | .572 | .578 | .592 | .589 | .753 |
| bbh_date_qwen8b | 250 | .912 | 22 | .740 | .726 | .660 | **.775** | .821 |
| bbh_date_ministral | 237 | .869 | 31 | .306 | .312 | .552 | **.645** | .841 |
| **macro** | | | | 0.578 | 0.584 | 0.608 | **0.685** | 0.791 |

**ChainUQ − CONV+FINAL = +0.076 macro, wins 4/5.** ChainUQ is the best single-pass
estimator on non-math too — it does **not** depend on a math answer parser. SC@8
(8×) remains stronger, consistent with the math story. **→ the method's scope is
general reasoning, not just mathematics; the title need not be math-restricted.**

### Layer 2 — math → non-math zero-shot transfer
Head trained ONLY on 13 math cells, applied to non-math: macro ChainUQ 0.600, Δ vs
best base **−0.029**, Δ>0 only **2/5** (positive on GPQA science: Qwen +0.019, Phi
+0.051; negative on BBH). **Verdict:** in-domain non-math generalization holds;
**do NOT claim math→non-math zero-shot transfer** — report as an honest limitation
(domain gap between math and BBH-style symbolic tasks).

## P0-5 — System efficiency (`experiments/system_efficiency.py`)
Real wall-clock on one B200, Qwen3-8B on math500 (n=64), median latency + throughput.
Output `system_efficiency_q8b_math500.json/.jsonl`.

| method | latency (median) | rel | throughput (q/s) |
|---|:-:|:-:|:-:|
| primary (1 trace) | 1.205 s | 1.00× | 0.83 |
| p_true | 1.208 s | 1.00× | 0.83 |
| **ChainUQ M=8** | **1.239 s** | **1.03×** | 0.81 |
| ChainUQ M=8, cache OFF | 1.370 s | 1.14× | 0.73 |
| SC@4 | 1.565 s | 1.30× | 0.64 |
| SC@8 | 2.091 s | 1.74× | 0.48 |

**Real measured overhead of ChainUQ M=8 = 1.03× a single generation** (probe adds
~39 ms on the shared KV cache) — vs SC@8 at **1.74×**. Prefix caching is load-bearing:
disabling it raises the probe cost 4× (39→167 ms → 1.14× total), empirically
validating the KV-cache-reuse design. This replaces the earlier decoded-token-ratio
claim (1.4–3.8%) with an honest wall-clock number.

*Caveats:* (1) `peak_gpu_gb` field reads 0 — `torch.cuda.max_memory_allocated()`
misses vLLM's caching allocator; use `nvidia-smi` peak for the paper's memory column.
(2) p_true ≈ 1.00× here because its verify prompt is truncated to a short forward;
a full-context verify pass would cost more. (3) A second point on long-trace
`olympiad` (max_tokens 8192) is running to show overhead shrinks as traces lengthen.
