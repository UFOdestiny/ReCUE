# ChainUQ — Final Results Dossier (source of truth for the KDD paper)

Everything the paper needs: experimental setup, every table with exact numbers copied
from `$EXP_ROOT` JSON caches, and the honest findings/verdicts. All UQ is **judge-free**
(deterministic `math_verify` for math; exact letter-match for MC). Companion machine
files: the JSONs named in each section. Positioning narrative options are in
`docs/STORY_OPTIONS.md`.

---

## 0. Experimental setup (frozen protocol)

- **Task.** Given ONE reasoning trace, predict whether its final answer is correct
  (binary correctness ranking). Metric: **AUROC** (rank correct > wrong); AURC/ECE where noted.
- **No judge model.** Math labels = `math_verify` (symbolic+numeric). MC labels = option-letter exact match.
- **Matrix.** 7 model families × 5 math datasets = **31 cells**. k=8 samples/question.
  - Models: Qwen3-4B / 8B / 14B, Qwen3.5-9B, Phi-4-reasoning, Ministral-3-14B-Reasoning, Llama-3.1-8B (MATH500 only).
  - Datasets: GSM8K, MATH500, Minerva, OlympiadBench, AMC23.
- **ChainUQ signal.** From ONE generation: cut the `<think>` trace at reasoning-step
  boundaries; at each cut force-decode `\boxed{...}` (KV-cache reused → cheap). Features:
  - CONV (6): answer-identity convergence — agree_frac, last_half_agree, final_stable_run, conv_frac, flip_rate, n_distinct.
  - CDYN (6): answer-CONFIDENCE dynamics — mean/last/min forced-answer first-token logprob, slope, first-agree conf, std.
  - SEQ (2): mean token logprob, mean token entropy.
- **Head.** Logistic regression (StandardScaler), **5-seed × 5-fold CV**, strict
  per-question separation. Logistic is the default (simple; answers the "patchwork" critique).
- **Cost unit.** 1× = one full generation. ChainUQ probes reuse the KV cache (measured 1.03×, §7).
  P(True) = +1 forward. SC@k = k× generations.
- **Compute.** 2× B200. Caches in `$EXP_ROOT`; per-cell OOF scores reproducible from `experiments/`.

---

## 1. MAIN TABLE — cost-tiered comparison (`main_table_v2.json`, `master_table.json`)

Methods placed in fair cost blocks. Reviewers reject cross-tier ranking, so we never
compare a 1× method against 8× SC in the same ranking. Macro AUROC over 31 cells.
Contrasts use **cell-equal hierarchical bootstrap** (2000 resamples; each cell equal
weight so GSM8K's large n cannot dominate).

| Cost tier | Method | Macro AUROC | #best-in-tier | Pre-registered contrast |
|---|---|:-:|:-:|---|
| **1× single-trace** | DeepConf-bottom | 0.592 | — | |
| | answer-convergence (prior-art) | 0.645 | — | |
| | mean log-prob | 0.682 | — | |
| | self-certainty | 0.684 | — | |
| | **ChainUQ (ours)** | **0.791** | **25/31** | **T1: ChainUQ − best-1× = +0.103, 95% CI [+0.056, +0.146], p<0.001 ✓** |
| **+1 forward** | P(True) | 0.755 | — | |
| | **ChainUQ + P(True)** | **0.821** | — | **T2: vs P(True) = +0.066, CI [+0.008, +0.122], p=0.011 ✓** |
| **8× sampling** | self-consistency@2 | 0.762 | — | |
| | self-consistency@4 | 0.859 | — | |
| | self-consistency@8 (SOTA ref) | 0.882 | — | |
| | **ChainUQ ⊕ SC@8 (fusion)** | **0.898** | — | **T3: vs SC@8 = +0.016, CI [+0.002, +0.031], p=0.013 ✓** |

**Findings.**
- **Within the true 1× tier, ChainUQ is SOTA and significantly so** (+0.103, p<0.001;
  best on 25/31 cells). Strict-1× per-cell margin over the best *other* 1× baseline:
  mean +0.035, wins 25/31, worst cell −0.229 (reported, not hidden).
- 1× **cannot** reach SC@8 (0.791 vs 0.882) — structural (single gen < 8× sampling). We do NOT claim it.
- **Matched-8×-budget fusion beats SC@8** (0.898 vs 0.882, +0.016, p=0.013, wins 24/31) — the defensible "beats SOTA" claim.
- P(True) is a cheap complementary boost (still needs its own +1-forward tier).

**Robustness of the headline (`stats_upgrade.json`):** macro / macro-without-AMC23 /
worst-dataset — ChainUQ 0.791 / 0.783 / 0.458; SC@8 0.882 / 0.866 / 0.702; fusion
0.898 / 0.888 / 0.736. Removing AMC23 (small n) barely moves anything.

Per-cell full 31-cell matrix (all 9 methods) is in Appendix Table A (source
`master_table.json`, reproduced in §11).

---

## 2. BIG ABLATION — is the signal trajectory dynamics or just the endpoint? (`ablation_order.json`, `traj_ablation.json`)

Two nested ablations, 31 cells, 5-seed CV, paired problem-level bootstrap.

### 2A. Endpoint vs trajectory (the novelty test) — Block A of the paper
| Configuration | Macro AUROC | Controls the alternative explanation |
|---|:-:|---|
| FINAL (final-probe confidence only) | 0.629 | "just the last confidence" |
| CONV (answer-convergence only) | 0.645 | "just answer-identity stability" (prior art) |
| CONV+FINAL (endpoint null) | 0.695 | "convergence + final logprob" |
| BAG-of-probes (order-invariant multiset) | 0.701 | "a static set of probe values is enough" |
| ORD-confidence trajectory | 0.730 | pre-final confidence dynamics |
| DUAL ordered trajectory (id+conf) | 0.743 | identity/confidence complementarity |
| **FULL (DUAL + sequence feats)** | **0.785** | complete method |

**Key contrasts:** FULL − CONV+FINAL = **+0.090, significant 18/31** (trajectory adds
signal beyond the endpoint). DUAL − BAG = **+0.042, significant 11/31** (order helps
over the unordered multiset).

### 2B. Order-invariance finding (honest, shapes the naming)
| control | Macro AUROC |
|---|:-:|
| DUAL (ordered) | 0.743 |
| PERM (train-perm → test-perm, 10 seeds) | 0.741 |
| REVERSE (time-reversed) | 0.754 |

DUAL − PERM = **+0.003 (sig only 3/31)**. **Destroying temporal order barely hurts once
retrained.** → The gain is a *multi-prefix commitment* distribution, NOT strict temporal
order. **Naming decision: call it "multi-prefix answer-commitment," not "temporal
dynamics."** Keep ordered features (they beat BAG), do not over-claim order-sensitivity.

### 2C. Named-component decomposition (also from `ablation_full.json`)
confidence-dynamics over answer-convergence: **+0.075 macro, sig 14/31**; per-family all
positive (Qwen3-8B +0.132, Ministral +0.120, Qwen3-14B +0.106, Qwen3-4B +0.040, Phi-4
+0.036, Qwen3.5 +0.017, Llama +0.061).

---

## 3. TRANSFER — does the signal generalize? (`transfer_lodo/lomo/global.json`, `transfer_head_capacity.json`)

Head trained on SOURCE cells, tested on UNSEEN target; normalization/features/classifier
fit on source only; no dataset-id/accuracy/target stats enter the head.

| Setting | Result | Verdict |
|---|---|---|
| **LODO** (leave-one-dataset-out, per backbone) | macro **0.796**, worst 0.550, degradation vs in-domain **−0.004**, ChainUQ>best-baseline **22/30** (10 sig) | **Strong cross-dataset transfer** — refutes "per-cell feature engineering" |
| **Global unified head** (no cell-id) | LOCO macro **0.767** = +cell-id upper bound 0.765 (per-cell 0.791) | **One shared head works**; cell identity adds nothing |
| **LOMO** (leave-one-model-family-out) | Qwen3 +0.025✓, size 4B/8B→14B +0.042✓; to-Ministral −0.065, to-Qwen3.5 −0.068 | **Mixed** — claim cross-dataset, NOT universal cross-model |

**Lever transfer verification (`transfer_head_capacity.json`, LODO):**
| lever | in-domain Δ | LODO transfer Δ | keep? |
|---|:-:|:-:|---|
| +P(True) feature | +0.038 | **+0.042 (21/25)** | **yes — robust, transfers** |
| RandomForest head | +0.035 | +0.015 (18/30) | cautious — ~60% is in-domain overfit |

---

## 4. MECHANISM — self-consistency's structural blind spot (`amp_stress.json`, `highconf_all.log`)

Pre-registered vote-fraction thresholds. On high-agreement subsets SC-confidence is
~constant → SC-AUROC → 0.5 by construction; ChainUQ still ranks correct vs wrong.

| SC vote ≥ | cells | total n | #wrong | SC AUROC | **ChainUQ AUROC** | fusion |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.625 | 26 | 9874 | 948 | 0.788 | 0.738 | 0.818 |
| 0.750 | 26 | 9145 | 591 | 0.737 | 0.703 | 0.770 |
| 0.875 | 24 | 7816 | 340 | 0.594 | 0.660 | 0.664 |
| **1.000 (unanimous)** | 24 | 7138 | **244** | **0.500** | **0.637** | 0.590 |

**This is the mechanistic reason the 8× fusion beats SC@8:** ChainUQ covers the
confident-consensus errors self-consistency is blind to (≥0.65 on 11/24 cells at unanimity).

---

## 5. GENERALIZATION beyond math (`nonmath_indomain.json`, `nonmath_transfer.json`)

12 new judge-free MC cells: BBH logical/tracking/date + GPQA-diamond × {Qwen3-8B,
Phi-4-reasoning, Ministral}. BBH logical/tracking saturate (0–9 errors → excluded);
evaluation carried by GPQA-diamond (acc 0.51–0.58, 83–97 errors) + bbh_date.

- **In-domain (5 cells ≥15 errors):** ChainUQ macro **0.685** vs CONV+FINAL 0.608
  (**+0.076, wins 4/5**), best single-pass (logprob 0.578, self-cert 0.584); SC@8 0.791.
  → ChainUQ is **not** math-parser-dependent; scope = general reasoning.
- **math→non-math zero-shot transfer:** macro 0.600, Δ vs best base −0.029, Δ>0 only 2/5.
  → **Report as honest limitation** — do NOT claim math→non-math transfer.

---

## 6. AMPLIFIERS (`amp_labeleff.json`, `amp_capacity.json`, `cue_ensemble.json`)

- **Label efficiency:** ChainUQ reaches **95% of full-data AUROC at ~25% labels**; at
  **1% labels (0.736) it already beats CONV+FINAL at 100% (0.737)**. Curves vs {1,2,5,10,25,50,100}%.
- **Classifier-capacity control:** logistic 0.789 / RF 0.817 / GBT 0.799 / MLP 0.752.
  Heads close → gain is the OBSERVATION, not capacity. (⚠ tree-boosting AUROC unreliable
  under 25-fold averaging; keep logistic.)
- **Cue robustness (P1-4):** 3 semantically-equivalent probe cues → AUROC spread only
  **0.018** (worst 0.691 vs mean 0.698). Prompt-cue invariant. Cue-ensemble gives no
  performance gain (report as robustness, not a lever).

---

## 7. EFFICIENCY — real wall-clock (`system_efficiency_q8b_math500.json`, `..._olympiad.json`)

Qwen3-8B, one B200, median latency (n=64).

| method | latency (median) | rel. | throughput (q/s) |
|---|:-:|:-:|:-:|
| primary (1 trace) | 1.205 s | 1.00× | 0.83 |
| P(True) | 1.208 s | 1.00× | 0.83 |
| **ChainUQ M=8** | **1.239 s** | **1.03×** | 0.81 |
| ChainUQ M=8, cache OFF | 1.370 s | 1.14× | 0.73 |
| SC@4 | 1.565 s | 1.30× | 0.64 |
| SC@8 | 2.091 s | 1.74× | 0.48 |

**Measured overhead of ChainUQ = 1.03×** a single generation (probe +39 ms on shared KV
cache), vs SC@8 at 1.74×. Prefix caching is load-bearing (off → probe cost 4×). Replaces
the old decoded-token ratio (1.4–3.8%) with an honest wall-clock number.
*Caveats:* peak-GPU-mem field unreliable (torch allocator misses vLLM) — use nvidia-smi;
P(True) ≈1.00× only because its verify prompt is short-truncated.

---

## 8. NEGATIVE / EXPLORATORY (kept for honesty)

- **Conformal selective prediction (`conformal_*.json`):** in-domain risk control works
  (validity ~1.0, ChainUQ best-coverage among 1× at fixed risk). BUT under domain shift
  ALL methods' guarantee collapses (validity ~0.45), and domain-robust strategies push
  coverage to ~0 — cross-domain conformal is infeasible here (heterogeneous dataset
  difficulty). **Dropped as a main claim.**
- Per-sample confidence-aware voting; rich-SC; accuracy-oriented budget allocation;
  doubt-injection — all weak/negative, recorded in `docs/EXPERIMENTS.md` §10.

---

## 9. Pre-registered primary contrasts (final, for the paper's stats paragraph)

| # | Contrast | Δ | 95% CI (hier. bootstrap) | p | verdict |
|---|---|:-:|:-:|:-:|:-:|
| T1 | ChainUQ − best 1× single-trace | +0.103 | [+0.056, +0.146] | <0.001 | ✓ SIG |
| T2 | ChainUQ+P(True) − P(True) | +0.066 | [+0.008, +0.122] | 0.011 | ✓ SIG |
| T3 | fusion − SC@8 | +0.016 | [+0.002, +0.031] | 0.013 | ✓ SIG |
| A1 | FULL − CONV+FINAL (trajectory adds) | +0.090 | (14–18/31 cell-sig) | — | ✓ |
| A2 | DUAL − PERM (order per se) | +0.003 | (3/31 cell-sig) | — | ✗ (→ rename) |

Also: Holm correction over 31 secondary cells retained; multi-seed generation variance
±0.009–0.035 (`variance_final.log`).

---

## 10. Claim discipline (what we say / do not say)

SAY: (1) best 1× single-trace UQ, significantly (+0.103); (2) matched-8× fusion beats
SC@8 (+0.016, sig) by covering SC's confident-error blind spot; (3) transfers across
datasets + single global head; (4) works on non-math reasoning; (5) 1.03× real cost;
(6) cheap +P(True) boost that transfers.
DO NOT SAY: 1× beats SC@8; temporal ORDER is essential (it's commitment distribution);
math→non-math zero-shot transfer; universal cross-model calibration; conformal guarantee
under shift; RF gain is domain-general.

---

## 11. Appendix Table A — full 31-cell matrix (from `master_table.json`)

Columns: mean_logprob | self_certainty | deepconf_bottom | p_true | answer_convergence | **ChainUQ** | SC@2 | SC@4 | SC@8

(cell | n | acc | ...) — exact values:

```
amc23_ministral   40 .500 | .733 .710 .708 .657 .575 .930 .777 .959 .921
amc23_phi4r       40 .925 | .757 .739 .252 .559 .991 1.000 .973 1.000 .991
amc23_qwen14b     40 .900 | .632 .618 .688 .722 .000 .458 .833 .979 .990
amc23_qwen35_9b   40 .450 | .492 .510 .462  -   .851 .823 .674 .765 .838
amc23_qwen4b      40 .875 | .903 .909 .977 .869 .526 .846 .757 .957 .971
amc23_qwen8b      40 .775 | .778 .760 .692 .731 .434 .871 .857 .968 .966
gsm8k_ministral  796 .916 | .564 .567 .363 .774 .449 .508 .773 .862 .839
gsm8k_phi4r      800 .964 | .823 .843 .366 .714 .613 .830 .735 .793 .820
gsm8k_qwen14b    800 .965 | .886 .890 .798 .754 .539 .879 .655 .779 .811
gsm8k_qwen35_9b  800 .896 | .863 .864 .719  -   .692 .910 .755 .860 .908
gsm8k_qwen4b     800 .945 | .892 .887 .831 .783 .670 .908 .775 .829 .894
gsm8k_qwen8b     800 .950 | .904 .905 .811 .786 .613 .913 .705 .814 .823
math500_llama8b  500 .472 | .659 .641 .649  -   .512 .693 .809 .870 .872
math500_ministral491 .617| .506 .509 .427 .778 .660 .709 .807 .908 .914
math500_phi4r    500 .912 | .565 .611 .275 .806 .806 .849 .897 .962 .974
math500_qwen14b  500 .824 | .750 .746 .655 .868 .628 .817 .862 .934 .968
math500_qwen35_9b500 .584 | .537 .535 .451  -   .810 .841 .711 .840 .885
math500_qwen4b   500 .772 | .706 .705 .693 .848 .792 .858 .806 .900 .936
math500_qwen8b   500 .792 | .696 .687 .624 .857 .614 .784 .825 .894 .943
minerva_ministral264 .367| .515 .513 .494 .681 .551 .571 .645 .765 .755
minerva_phi4r    272 .460 | .497 .520 .452 .656 .659 .678 .669 .719 .731
minerva_qwen14b  272 .504 | .602 .594 .537 .736 .562 .619 .646 .749 .791
minerva_qwen35_9b272 .335 | .551 .546 .538  -   .734 .770 .602 .649 .702
minerva_qwen4b   272 .397 | .609 .604 .585 .756 .669 .729 .713 .818 .834
minerva_qwen8b   272 .478 | .591 .590 .577 .713 .603 .653 .663 .746 .774
olympiad_ministral579 .383| .641 .641 .623 .705 .657 .788 .771 .887 .921
olympiad_phi4r   581 .707 | .599 .651 .395 .677 .696 .864 .807 .890 .900
olympiad_qwen14b 581 .656 | .749 .751 .690 .818 .726 .822 .786 .913 .941
olympiad_qwen35_9b581 .358| .666 .682 .599  -   .857 .889 .729 .806 .831
olympiad_qwen4b  581 .627 | .739 .736 .703 .799 .749 .852 .794 .917 .941
olympiad_qwen8b  581 .616 | .737 .737 .735 .828 .745 .847 .803 .906 .948
```

---

## 12. Script → output index (reproducibility)

| result | script | output |
|---|---|---|
| main cost-tiered table + T1/T2/T3 | `experiments/main_table_v2.py` | `main_table_v2.json` |
| full 31-cell matrix | `experiments/main_comparison.py` | `master_table.json` |
| order ablation (Block A) | `experiments/ablation_temporal_order.py` | `ablation_order.json` |
| endpoint ablation | `experiments/ablation_trajectory.py` | `traj_ablation.json` |
| component ablation | `experiments/ablation_confidence.py` | `ablation_full.json` |
| transfer LODO/LOMO/global | `experiments/transfer.py` | `transfer_{lodo,lomo,global}.json` |
| lever transfer verification | `experiments/transfer_head_capacity.py` | `transfer_head_capacity.json` |
| SC blind-spot stress | `experiments/novelty_amplifiers.py --exp stress` | `amp_stress.json` |
| label efficiency / capacity | `experiments/novelty_amplifiers.py` | `amp_{labeleff,capacity}.json` |
| cue robustness | `scripts/run_probe_multicue.py` + `experiments/cue_ensemble.py` | `cue_ensemble.json` |
| non-math generalization | `experiments/non_math_generalization.py` | `nonmath_{indomain,transfer}.json` |
| efficiency | `experiments/system_efficiency.py` | `system_efficiency_q8b_*.json` |
| hierarchical stats | `experiments/stats_upgrade.py` | `stats_upgrade.json` |
| performance levers | `experiments/perf_levers.py` | `perf_levers.json` |

---

## 8b. Adaptive budget allocation frontier (NEGATIVE, `adaptive_frontier.json`)

Tested whether ChainUQ-guided escalation (spend extra samples on the lowest-confidence
rho fraction) beats uniform SC and random routing on the AUROC-vs-avg-cost frontier
(31 cells, k=8). Result at scale is negative and confirms the earlier accuracy-routing
dead end:
- vs RANDOM routing: ChainUQ routing is **worse** by 0.006-0.010 across the frontier
  (it escalates genuinely hard/unrankable questions, wasting budget). The routing
  DECISION adds no value.
- vs UNIFORM SC: mixed, not dominant (+0.028 at 2.4x but -0.014 at 3.8-4.5x); reaches
  SC@8 quality only at 8x (no saving).
- The full-fusion endpoint (8x, 0.898) still beats SC@8 (0.882) by +0.016, but that is
  the matched-budget fusion result (RQ3), not an allocation gain.
**Verdict: DROP adaptive allocation as a claim.** Keep matched-budget fusion only. The
6-cell pilot looked positive (reached SC@8 at 4.5x) but was small-sample optimistic.
