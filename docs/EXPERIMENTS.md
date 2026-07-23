# TRACE — Full Experimental Dossier

Detailed, source-traceable record of all experiments behind **TRACE**
(*Trajectory of Answer-Confidence Estimation*): judge-free uncertainty
quantification (UQ) for reasoning LLMs. Companion to `docs/RESULTS.md` (which is
the condensed narrative); this file is the exhaustive tables + story + baseline
map + novelty analysis. Every number is copied from a log under `$EXP_ROOT`; the
source log is named in each section.

- **Task.** Given one reasoning trace, predict whether its final answer is correct.
- **Labels.** Deterministic `math_verify` (symbolic+numeric). No judge model anywhere.
- **Metric.** AUROC (rank correct>wrong); AURC / ECE where noted. Supervised heads use
  5-seed 5-fold CV with strict per-question separation.
- **Matrix.** 7 model families x 5 datasets = 31 cells.
  Models: Qwen3-4B/8B/14B, Qwen3.5-9B, Phi-4-reasoning, Ministral-3-14B-Reasoning, Llama-3.1-8B.
  Datasets: GSM8K, MATH500, Minerva, OlympiadBench, AMC23. k=8 samples per question.
- **Cost unit.** 1x = one full trace generation; SC@k = kx. TRACE probes reuse the KV cache.

---

## 0. The story in one line

> Correct and incorrect reasoning traces differ in **how the model's answer
> confidence evolves** along the trace. This single-generation signal (i) beats
> every same-cost UQ baseline, (ii) is provably *trajectory* information (not the
> endpoint), and (iii) is **complementary** to self-consistency because it detects
> the confident-consensus errors that answer-agreement is structurally blind to.

Three-layer contribution:
1. **1x cost** — best single-generation UQ (Sec. 1).
2. **evidence it is real & novel** — trajectory ablation (Sec. 3) + mechanism (Sec. 4)
   + robustness (Sec. 5–7).
3. **beats SOTA** — matched-budget fusion with self-consistency (Sec. 2).

---

## 1. Main comparison: single-generation UQ (RQ-Performance)

**Macro AUROC over all 31 cells** (`master_table.log`). `#best` = cells where the
method is the top single-pass estimator.

| method | cost | macro AUROC | #best (of single-pass) |
|---|---|:---:|:---:|
| DeepConf-bottom | 1x | 0.592 | 1 |
| answer-convergence (prior-art proxy) | 1x | 0.645 | 1 |
| mean log-prob | 1x | 0.682 | 0 |
| self-certainty | 1x | 0.684 | 2 |
| P(True) | 1 fwd | 0.755 | 9 |
| **TRACE (ours)** | **1x** | **0.791** | **18** |
| self-consistency@2 | 2x | 0.762 | — |
| self-consistency@4 | 4x | 0.859 | — |
| self-consistency@8 | 8x | 0.882 | — |

**Read.** TRACE is the best single-generation estimator on 18/31 cells (next: P(True), 9),
macro 0.791. It beats every same-cost baseline and matches SC@2–4 while single-pass.
It does **not** match SC@8 (0.791 vs 0.882) at 1x — we do not claim that.

**TRACE minus answer-convergence, per family** (`master_table.log`); positive on all 7:

| family | ΔAUROC | | family | ΔAUROC |
|---|:---:|---|---|:---:|
| Qwen3-14B | +0.228 | | Qwen3-4B | +0.158 |
| Qwen3-8B | +0.212 | | Ministral | +0.123 |
| Llama-3.1-8B | +0.181 | | Phi-4-reasoning | +0.091 |
| | | | Qwen3.5-9B | +0.058 |
| | | | **OVERALL** | **+0.146** |

**Significance vs the best same-cost (1x) baseline** (paired example bootstrap,
`sig_vs1x.log`): TRACE wins 9/10 head-to-head cells, **significant 8/10** (Δ up to
+0.222 on amc23-Qwen8B; +0.133 olympiad-Qwen4B; +0.099 math500-Qwen8B). Only
minerva-Qwen8B is a (non-significant) loss.

### Per-cell detail (fullcmp_*.log), AUROC, cost in header

**Qwen3-4B**

| method | cost | gsm8k | math500 | minerva | olympiad | amc23 | AVG |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| mean_logprob | 1x | 0.892 | 0.706 | 0.609 | 0.739 | 0.903 | 0.770 |
| self_certainty | 1x | 0.887 | 0.705 | 0.604 | 0.736 | 0.909 | 0.768 |
| deepconf_bottom | 1x | 0.831 | 0.693 | 0.585 | 0.703 | 0.977 | 0.758 |
| P(True) | 1fwd | 0.783 | 0.848 | 0.756 | 0.799 | 0.869 | 0.811 |
| **TRACE (ours)** | 1x | **0.912** | **0.917** | **0.823** | **0.932** | 0.966 | **0.910** |
| self_consistency@8 | 8x | 0.894 | 0.936 | 0.834 | 0.941 | 0.971 | 0.915 |

**Qwen3-8B**

| method | cost | gsm8k | math500 | minerva | olympiad | amc23 | AVG |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| mean_logprob | 1x | 0.904 | 0.696 | 0.591 | 0.737 | 0.778 | 0.741 |
| self_certainty | 1x | 0.905 | 0.687 | 0.590 | 0.737 | 0.760 | 0.736 |
| P(True) | 1fwd | 0.786 | 0.857 | 0.713 | 0.828 | 0.731 | 0.783 |
| **TRACE (ours)** | 1x | **0.932** | **0.956** | 0.696 | **0.942** | **1.000** | **0.905** |
| self_consistency@8 | 8x | 0.823 | 0.943 | 0.774 | 0.948 | 0.966 | 0.891 |

Note: on Qwen3-8B single-generation TRACE (0.905) already exceeds SC@8 (0.891) on
average — driven by gsm8k where SC@8 is weak (0.823).

**Ministral-3-14B-Reasoning** (cross-family; note logprob baselines near-random)

| method | cost | gsm8k | math500 | minerva | olympiad | amc23 | AVG |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| mean_logprob | 1x | 0.588 | 0.529 | 0.537 | 0.643 | 0.733 | 0.606 |
| self_certainty | 1x | 0.592 | 0.532 | 0.536 | 0.643 | 0.710 | 0.602 |
| deepconf_bottom | 1x | 0.399 | 0.453 | 0.518 | 0.625 | 0.708 | 0.540 |
| **TRACE (ours)** | 1x | **0.866** | **0.918** | 0.731 | **0.915** | 0.887 | **0.863** |
| self_consistency@8 | 8x | 0.810 | 0.897 | 0.746 | 0.927 | 0.921 | 0.858 |

**Cross-family highlight.** On Ministral, all logprob/self-certainty/DeepConf baselines
are 0.54–0.61 (near chance), yet TRACE holds 0.863 and single-generation TRACE again
edges SC@8 (0.858). This shows the signal is not a repackaging of token logprob.

---

## 2. Beats SOTA at matched budget (RQ-BeatsSOTA)

Two ways TRACE surpasses self-consistency when a larger budget is allowed.

### 2a. Matched 8x fusion — TRACE features + SC@8 statistics (`beat_sota.log`)

Fuse TRACE's single-chain features with SC@8's vote+entropy at the **same 8x budget**;
compare to SC@8 alone. Paired bootstrap CI.

- **Mean AUROC(fusion − SC@8) = +0.022; significant in 10/25 cells; never significantly
  worse except 2 easy Ministral cells.**
- Largest wins exactly where SC is unreliable:

| cell | SC@8 | fusion | Δ | p |
|---|:-:|:-:|:-:|:-:|
| gsm8k Qwen3-8B | 0.823 | **0.941** | +0.118 | 0.000 |
| gsm8k Qwen3-14B | 0.811 | **0.910** | +0.101 | 0.001 |
| olympiad Qwen3.5-9B | 0.831 | **0.920** | +0.089 | 0.000 |
| minerva Qwen3.5-9B | 0.702 | **0.772** | +0.070 | 0.015 |
| gsm8k Phi-4-reasoning | 0.820 | **0.874** | +0.055 | 0.008 |
| olympiad Phi-4-reasoning | 0.900 | **0.924** | +0.023 | 0.002 |
| olympiad Qwen3-4B | 0.941 | **0.952** | +0.011 | 0.024 |
| math500 (all, SC@8≈0.94) | ~0.94 | ~0.94 | ≈0 (ns) | — |

`dynWvote` (confidence-weighted voting, no learned fusion) alone ≈ SC@8: the gain
comes from the **fusion**, i.e. the confidence-dynamics signal is *complementary* to
cross-sample agreement, not redundant.

### 2b. Higher-cost hybrid — TRACE(1x) + a few SC samples (`hybrid_uq.log`)

hybrid@4 = TRACE + 4 samples (~4x) vs SC@8 (8x): **mean +0.017 AUROC, significant in
6/25 cells, never significantly worse except math500-Qwen8B (−0.026).** Reaches / beats
8-sample SC quality at half the samples on GSM8K and weak-logprob models.

Per-family averages (from `fullcmp_*.log`): hybrid@2(~2.1x) vs SC@8(8x): Qwen4B
0.920 vs 0.915; Qwen8B 0.918 vs 0.891; Ministral 0.885 vs 0.858. hybrid@4(~4.1x):
Qwen4B 0.933, Qwen8B 0.924, Ministral 0.920.

---

## 3. Ablation: the signal is trajectory DYNAMICS, not the endpoint (RQ-Ablation)

The decisive novelty test (`traj_ablation.log`, 31 cells, 5-seed CV, bootstrap CI).
Nested feature groups:
- **FINAL** = final-probe answer confidence only.
- **CONV** = answer-identity convergence (the prior-art content).
- **C+F** = CONV + FINAL (the null hypothesis: "answer-convergence + final logprob").
- **FULL** = CONV + FINAL + confidence **trajectory**.
- **C+TRAJ** = CONV + trajectory, **final value removed**.

| configuration | macro AUROC |
|---|:---:|
| FINAL (final-probe confidence only) | 0.629 |
| CONV (answer-convergence only) | 0.645 |
| C+F (CONV + FINAL) — null hypothesis | 0.696 |
| **FULL (CONV + FINAL + trajectory)** | **0.742** |
| **C+TRAJ (trajectory, final removed)** | **0.739** |

**Two clinching facts.**
1. FULL − C+F = **+0.046 macro, significant in 14/31 cells** (up to +0.19 on GSM8K):
   trajectory shape adds signal beyond convergence + final confidence.
2. Removing the final value entirely (C+TRAJ 0.739) barely changes AUROC and stays far
   above C+F (0.696): the information lives in the **trajectory**, not the endpoint.

This directly refutes "TRACE = answer-convergence + final-answer log-probability."

**Component ablation (confidence-dynamics over answer-convergence, `ablation_full.log`).**
+0.075 macro, significant 14/31; per-family all positive (Qwen3-8B +0.132, Ministral
+0.120, Qwen3-14B +0.106, Qwen3-4B +0.040, Phi-4 +0.036, Qwen3.5 +0.017, Llama +0.061).

---

## 4. Mechanism: why we beat SOTA (RQ-Mechanism)

**Confidence trajectories** (`figs/mechanism_trajectories.png`, `visualize.log`).
Correct traces rise to and plateau at high forced-answer confidence (final logprob
mean −0.041); wrong traces stay in a separated lower band (−0.231); the bands do not
overlap. The answer-agreement curves separate too but **collapse together at the trace
end** (both →1.0), so convergence is discriminative only mid-trace while confidence
stays discriminative throughout.

**Self-consistency's structural blind spot** (`highconf_all.log`). On the
**high-agreement subpopulation** (SC vote ≥ 0.75), SC confidence is ~constant so its
AUROC ≈ 0.5 by construction; yet TRACE still ranks correct vs wrong:

| high-agreement subset | n | acc | SC AUROC | **TRACE AUROC** |
|---|:-:|:-:|:-:|:-:|
| gsm8k Qwen3-14B | 776 | .983 | ~0.5 | **0.817** |
| gsm8k Qwen3-8B | 767 | .975 | ~0.5 | **0.813** |
| gsm8k Phi-4-reasoning | 781 | .976 | ~0.5 | **0.808** |
| math500 Qwen3-8B | 365 | .962 | ~0.5 | **0.790** |
| olympiad Qwen3-4B | 353 | .929 | ~0.5 | **0.753** |
| olympiad Phi-4-reasoning | 368 | .927 | ~0.5 | **0.721** |
| olympiad Ministral | 176 | .909 | ~0.5 | **0.694** |

**Aggregate:** over 23 cells with n≥100, TRACE mean AUROC = **0.659** on the exact
subpopulation where SC is at chance (≥0.65 on 13/23). This is the mechanistic reason
the 8x fusion (Sec. 2a) wins: TRACE covers self-consistency's confident-error blind spot.

---

## 5. Confound control: not a length/difficulty detector (RQ-Robustness)

`confound.log`, 12 model-dataset cells, per-question bootstrap.

- Adding stability features on top of [logprob + entropy + **trace length**] lifts
  AUROC by **+0.059 macro** (positive in all 12 cells; up to +0.111 gsm8k-Ministral).
- Within-length-tercile AUROC (difficulty held ~constant by length) = **0.737 macro**
  → not explained by trace length.
- corr(stability, length) = −0.47 (moderate), corr(stability, logprob) = +0.17 (weak).
- Stronger control = **within-problem AUROC** (`within_problem.log`): on the *same
  problem*, ranking correct vs wrong traces, cross-sample agreement/stability = **0.791
  macro** while per-sample logprob = **0.522** (chance). Difficulty, gold, domain,
  wording are all held constant here.

---

## 6. Multi-seed stability (RQ-Robustness)

3 independent generation seeds (`variance_final.log`), AUROC mean±std.

| cell | logprob | P(True) | answer_conv | **TRACE** | SC@8 |
|---|---|---|---|---|---|
| math500 Qwen3-8B | .691±.008 | .857±.000 | .647±.019 | **.791±.009** | .940±.002 |
| math500 Ministral | .515±.010 | .778±.000 | .604±.055 | **.669±.035** | .909±.005 |
| olympiad Qwen3-8B | .739±.010 | .828±.000 | .733±.014 | **.835±.013** | .932±.012 |
| olympiad Ministral | .654±.015 | .705±.000 | .669±.008 | **.781±.017** | .880±.081 |

TRACE is stable across generation seeds (std .009–.035) and consistently beats
answer-convergence and logprob. On olympiad-Ministral it even beats P(True) while SC@8
is itself unstable (±.081).

---

## 7. Efficiency (RQ-Efficiency)

Probe cost is the marginal decoded tokens of Anchored Answer Probing over one full
generation (KV-cache reused; `probe_ntok` vs `n_gen_tokens`).

| cell | primary trace (tok) | probe overhead |
|---|:-:|:-:|
| gsm8k Qwen3-8B | ~1,807 | **+3.8%** |
| math500 Qwen3-8B | ~3,105 | **+2.5%** |
| olympiad Qwen3-8B | ~6,396 | **+1.4%** |

TRACE is thus ~**1.0x** generation cost (overhead shrinks as traces lengthen — best on
the expensive long-reasoning cases). P(True) needs one extra full forward pass;
SC@k needs kx full generations. Caveat: overhead is reported in decoded tokens; wall-clock
also depends on batching / KV-cache scheduling (see limitations).

---

## 8. Baseline map (what each competitor uses, and how TRACE differs)

| baseline | cost | signal | needs | TRACE difference |
|---|---|---|---|---|
| mean log-prob / perplexity | 1x | sequence token prob | logprobs | uses answer-confidence *trajectory*, not raw token prob |
| token entropy / SAR | 1x | per-token entropy | logprobs | same |
| self-certainty | 1x | output-dist peakedness | logprobs | same |
| DeepConf-tail/bottom | 1x | worst-window token conf | logprobs | probes the committed *answer*, not raw tokens |
| P(True) | +1 fwd | self-verdict prob | 1 extra pass | no extra pass; reads implicit answer, not a verbal verdict |
| answer-convergence (prior-art proxy) | 1x* | when the intermediate answer stabilizes | probes | adds *confidence* dynamics; ablation shows this is the source of gain |
| self-consistency@k | kx | cross-sample answer agreement | k gens | single generation; complementary (covers SC's confident-error blind spot) |
| semantic entropy@k | kx | meaning-cluster entropy | k gens + NLI | same |
| CISC / DeepConf-vote@k | kx | confidence-weighted vote | k gens + logprobs | same |

All baselines are judge-free and evaluated under identical labels/splits.

---

## 9. Novelty statement (defensible, and how it survives review)

**Claim.** TRACE is the first UQ method to model the **temporal dynamics of a model's
answer confidence along a single reasoning trace**, and to show this is (a) distinct
from answer-identity convergence, final-answer probability, and cross-sample agreement,
and (b) a covering signal for self-consistency's confident-error blind spot.

**Differentiation from the nearest prior work.**
- **Self-consistency / semantic entropy** — answer agreement across many samples; cost
  kx; blind to confident-consensus errors. TRACE: single trace; complementary at matched
  cost (Sec. 2a, Sec. 4).
- **Prefix-probing / early-exit methods (e.g. CGR, answer-convergence)** — use the
  *identity* of the intermediate answer and/or single-point certainty for early stopping.
  TRACE: uses the *confidence trajectory*, and the ablation (Sec. 3) shows the gain is
  from the trajectory shape, not the endpoint or convergence, so it is not subsumed.
- **Chain-of-Embedding / internal-state probes** — static hidden-state geometry, need
  white-box internals. TRACE: uses the behavioral force-decoded answer distribution.

**What we deliberately do NOT claim** (each was tested and would fail review):
- "1x TRACE matches SC@8" — false on average; we claim best *same-cost* + near SC@2–4.
- "TRACE beats SC@8 accuracy via allocation" — `budget_alloc.log` shows P(True) is a
  better accuracy router; we make no test-time-accuracy claim.
- "per-sample confidence-aware voting beats SC" — `fusion_frontier.log` /
  `conf_vote_*.log`: Fus@4−SC@8 = −0.014, Fus@8−SC@8 = +0.001 (1/6); once k full samples
  exist, agreement is redundant with per-sample confidence. Dead end, reported honestly.
- Statistical significance is claimed only where a paired bootstrap supports it.

---

## 10. Negative / exploratory results (kept for honesty and reproducibility)

| study | file | outcome |
|---|---|---|
| rich self-consistency (answer-distribution features) | `rich_sc.log` | does NOT beat SC@8 (−0.013, 2/25) |
| per-sample confidence-aware voting | `conf_vote_*.log` | fragile; does not beat SC@8 |
| learned per-sample fusion frontier | `fusion_frontier_partial.log` | Fus@4−SC@8 −0.014; Fus@8−SC@8 +0.001 (1/6) |
| accuracy-oriented budget allocation | `budget_alloc.log` | P(True) is a better router than ours |
| single/bidirectional doubt-injection | `novelty_abl*.log`, `bidir_*.log` | weak (~+0.02); dropped |
| Prefix-Consistency baseline (resampled continuations) | `prefix_*.log` | split: ours wins gsm8k, PC wins hard math; ~18x cost; not pursued |

**Takeaway.** The robust, defensible wins are: (1) best single-generation UQ, (2)
matched-budget fusion beating SC@8 via complementarity, (3) trajectory-not-endpoint
ablation. The exploratory directions are recorded so the main claims are not overstated.

---

## 11. Source-log index (traceability)

| result | source log/json |
|---|---|
| main comparison macro + per-family | `master_table.log`, `master_table.json` |
| per-cell comparison | `fullcmp_q4b.log`, `fullcmp_q8b.log`, `fullcmp_q14b.log`, `fullcmp_ministral.log` |
| significance vs 1x SOTA | `sig_vs1x.log`, `significance.log` |
| matched-budget fusion | `beat_sota.log` |
| higher-cost hybrid | `hybrid_uq.log` |
| trajectory ablation | `traj_ablation.log`, `traj_ablation.json` |
| component ablation | `ablation_full.log`, `ablation_full.json` |
| mechanism plots + numbers | `visualize.log`, `figs/mechanism_*.png` |
| high-agreement blind spot | `highconf_all.log` |
| confound / length control | `confound.log`, `within_problem.log` |
| multi-seed variance | `variance_final.log` |
| efficiency | `gen/*.json` (`n_gen_tokens`), `probe/*.json` (`probe_ntok`) |
| negative results | `rich_sc.log`, `conf_vote_*.log`, `fusion_frontier_partial.log`, `budget_alloc.log`, `prefix_*.log` |

Reproduce any table from cached tensors with the matching `experiments/` or
`analysis/` module; see `README.md`.
