# Answer-Confidence Dynamics for Judge-Free Reasoning Uncertainty
### Consolidated experimental results & interpretation

**Setting.** Judge-free, verifier-supervised uncertainty quantification (UQ) for
reasoning LLMs. Correctness labels come from `math_verify` (deterministic), never a
judge model. We probe a *single completed reasoning trace*: cut the `<think>` trace at
step boundaries and force-decode `The final answer is \boxed{...}` at each cut (reusing
vLLM prefix cache → ~2% extra generated tokens). This yields, per step, an intermediate
**answer** and the model's **forced-answer confidence** (first-token logprob).

**Method (ours).** A lightweight logistic head over three feature groups from one trace:
- **CONV** — answer-*identity* dynamics (does the intermediate answer converge to the
  final one, when, how consistently). This is the "answer-convergence" prior-art content.
- **CDYN / TRAJ** — answer-*confidence* dynamics (trajectory of the forced-answer logprob:
  level, slope, volatility, early-vs-late gap, area). **This is our novel contribution.**
- **SEQLP** — cheap sequence logprob/entropy (single-pass baseline features).

**Matrix.** 7 model families × 5 datasets = 31 cells. Models: Qwen3-4B/8B/14B, Qwen3.5-9B,
Phi-4-reasoning, Ministral-3-14B-Reasoning, Llama-3.1-8B. Datasets: GSM8K, MATH500,
Minerva, OlympiadBench, AMC23. Metrics: AUROC, AURC (selective risk), with 5-seed CV for
supervised heads and problem-level bootstrap CIs for key contrasts.

---

## Result 1 — Single-generation SOTA comparison (macro AUROC, 31 cells)

| method                       | cost  | macro AUROC | # best-of-single-pass |
|------------------------------|-------|:-----------:|:---------------------:|
| deepconf_bottom (Fu'25)      | 1×    | 0.592       | 1 |
| answer_convergence (prior)   | 1×    | 0.645       | 1 |
| mean_logprob                 | 1×    | 0.682       | 0 |
| self_certainty (Kang'25)     | 1×    | 0.684       | 2 |
| P(True) (Kadavath'22)        | 1 fwd | 0.755       | 9 |
| **ours (conv + confdyn)**    | 1×    | **0.791**   | **18** |
| self_consistency@2           | 2×    | 0.762       | — |
| self_consistency@4           | 4×    | 0.859       | — |
| self_consistency@8           | 8×    | 0.882       | — |

**Read.** Ours is the best single-generation method on **18/31 cells** (next: P(True), 9),
macro **0.791**, beating every same-cost baseline and matching SC@2–4 while single-pass.
Ours beats the answer-convergence prior-art baseline by **+0.146 macro**, positive on all
7 families (Qwen3-14B +0.228, Qwen3-8B +0.212, Llama +0.181, Qwen3-4B +0.158,
Ministral +0.123, Phi-4 +0.091, Qwen3.5 +0.058).

**Honest scope.** Ours does **not** match SC@8 (0.791 vs 0.882) at 1× cost. The correct
claim is: *best low-cost / single-generation UQ, competitive at a fraction of SC's cost.*

---

## Result 2 — DECISIVE: the signal is trajectory DYNAMICS, not the endpoint

Reviewer's key concern: is the gain just "answer-convergence + final-answer logprob"?
Nested ablation (31 cells, 5-seed CV, problem-level bootstrap):

| config                                       | macro AUROC |
|----------------------------------------------|:-----------:|
| FINAL (final-probe confidence only)          | 0.629 |
| CONV (answer-convergence only)               | 0.645 |
| C+F (conv + final confidence) — *null hyp.*  | 0.696 |
| **FULL (conv + final + trajectory)**         | **0.742** |
| **C+TRAJ (conv + trajectory, final removed)**| **0.739** |

**Read (two clinching facts).**
1. FULL − C+F = **+0.046 macro, significant in 14/31 cells** (up to +0.19 on GSM8K):
   trajectory shape adds real signal beyond convergence + final confidence.
2. Removing the final confidence entirely (**C+TRAJ = 0.739**) still ≈ FULL and far above
   C+F (0.696) → the discriminative information lives in the **trajectory**, not the
   endpoint. Directly refutes "just answer-convergence + final logprob."

Strongest on GSM8K and OlympiadBench (all significant). Weakest where the base head is
already near-ceiling (some MATH500/Minerva cells).

---

## Result 3 — Mechanism (see figs/mechanism_trajectories.png)

Forced-answer confidence vs normalized reasoning position, split by correctness:
- **Correct** traces rise to and plateau at **high** confidence (final logprob ≈ −0.04).
- **Wrong** traces stay in a **clearly separated lower band** throughout (≈ −0.23);
  the two confidence bands **never overlap**.
- The answer-convergence (agreement) curves also separate but **collapse together at the
  trace end** (both → 1.0, since the final answer is the reference) — so convergence is
  discriminative only mid-trace, while **confidence stays discriminative throughout**.

Caveat (honest): the confidence *slope* is nearly identical for correct/wrong
(+0.054 vs +0.058); the discriminator is the confidence **level trajectory** (and its
volatility / area), not the trend. Result 2 confirms the shape (excl. final) still adds.

---

## Result 4 — Cross-family robustness & significance

- **vs same-cost SOTA:** ours beats the best 1× competitor on **9/10** head-to-head
  cells, significant (paired bootstrap) on **8/10** (Δ up to +0.22).
- **Cross-family:** on Ministral-3-14B-Reasoning, all logprob-style 1× baselines are
  near-random (0.54–0.61) yet ours holds **0.86** — the signal captures something
  logprob-based UQ fundamentally misses.
- **Confidence-dynamics ablation gain** (CDYN over answer-convergence): +0.075 macro,
  significant 14/31 cells; per-family all positive (Qwen3-8B +0.132, Ministral +0.120,
  Qwen3-14B +0.106, …).

---

## Result 5 — Multi-generation-seed variance (3 independent decode seeds)

| cell               | logprob     | P(True) | answer_conv | **ours**    | SC@8       |
|--------------------|-------------|---------|-------------|-------------|------------|
| math500_qwen8b     | .691±.008   | .857    | .647±.019   | **.791±.009** | .940±.002 |
| math500_ministral  | .515±.010   | .778    | .604±.055   | **.669±.035** | .909±.005 |
| olympiad_qwen8b    | .739±.010   | .828    | .733±.014   | **.835±.013** | .932±.012 |
| olympiad_ministral | .654±.015   | .705    | .669±.008   | **.781±.017** | .880±.081 |

**Read.** Ours is stable across independent generation seeds (std .009–.035) and
consistently beats answer-convergence and logprob. On olympiad_ministral ours (.781)
> P(True) (.705) > answer_conv (.669) > logprob (.654), while SC@8 is itself unstable (±.081).

---

## Result 6 — Higher-cost variant (the honest "beats SOTA" claim)

hybrid@4 = ours(1×) + 4 self-consistency samples, fused by logistic (~4× cost) vs SC@8 (8×):

- Mean AUROC(hybrid@4 − SC@8) = **+0.017**; **significantly better in 6/25 cells**,
  never significantly worse except math500_qwen8b (−0.026).
- Wins concentrate on GSM8K (Qwen3-8B **+0.124**, Qwen3-14B +0.105, Phi-4 +0.069) and
  some OlympiadBench (Qwen3.5 +0.079); on MATH500 SC@8 is hard to beat (ties).

**Read.** A defensible "beats SOTA at half the cost" claim, stated **per-setting**:
hybrid@4 matches or exceeds 8-sample self-consistency's UQ quality using 4 samples, with
significant wins on ~1/4 of settings (esp. GSM8K and weak-logprob models).

**Negative result (reported honestly).** For *accuracy-oriented* budget allocation
(deciding who gets more samples to maximize final vote accuracy), P(True) is a better
router than our signal, and neither reliably beats uniform SC@k. Our contribution is
**UQ / selective-prediction quality**, not test-time-scaling accuracy — we do not claim
the latter.

---

## Result 7 — BEATS SOTA at matched cost: dynamics ⊕ self-consistency (25 cells, CI)

Fuse our answer-confidence-dynamics features with SC@8 statistics (vote + semantic
entropy), logistic head, at the **same 8× sample budget** as SC@8:

- Mean AUROC(ours⊕SC@8 − SC@8) = **+0.022**; **significantly better in 10/25 cells**,
  never significantly worse except 2 easy cells (gsm8k/minerva-Ministral, ≤ −0.026).
- Wins concentrate exactly where self-consistency is unreliable:
  GSM8K Qwen3-8B **+0.118**, Qwen3-14B **+0.101**, Phi-4 +0.055; OlympiadBench Qwen3.5
  **+0.089**, Minerva Qwen3.5 **+0.070**. Neutral where SC@8 already ≥ 0.94 (MATH500).
- `dynWvote` (confidence-weighted voting) alone ≈ SC@8 — the gain comes from the *fusion*,
  i.e. the confidence-dynamics signal is **complementary** to cross-sample agreement.

**Read (the defensible SOTA-beating claim).** Answer-confidence dynamics is a signal
**self-consistency does not capture**; fusing them improves UQ over 8-sample
self-consistency at matched cost, with significant gains on ~40% of settings — largest
precisely on easy datasets and weak-logprob models where SC's vote-agreement is
uninformative. This is the strongest "beats SOTA" evidence: same budget, better UQ,
clear mechanism (complementarity).

---

## Result 8 — where the beat-SOTA gain does (and does not) come from

Ablating the matched-budget (8×) fusion:
- **rich-SC** (6 features from the answer *distribution* of the same 8 samples: vote,
  top2-margin, neg-entropy, n-distinct, singleton-frac, none-frac) does **NOT** beat
  vanilla SC@8 (mean −0.013, sig in only 2/25). Vote fraction already captures the
  distribution; extra SC statistics overfit.
- **rich-SC + our confidence-dynamics** beats SC@8 (mean +0.024, **sig 11/25 cells**),
  matching Result 7 (+0.022, 10/25).
=> The beat-SOTA gain comes specifically from the **single-chain answer-confidence
dynamics being complementary to cross-sample agreement**, NOT from squeezing more out of
the sample distribution. Gain currently caps ~+0.024; largest where SC is weak
(gsm8k weak-logprob models, olympiad_qwen35 +0.028).

## Result 9 — confidence-AWARE voting does NOT beat SC (negative, reported honestly)

We probed the confidence trajectory of EACH of the k self-consistency samples and
tried confidence-weighted / learned-fusion voting at matched and reduced budgets:
- Confidence-weighted voting: helps at very low k (CA@2 − SC@2 ≈ +0.056 on olympiad) but
  is fragile (hurts on minerva_ministral) and never beats SC@8 with fewer samples.
- Learned fusion (per-sample confidence + SC@k + primary dynamics, logreg): Fus@4 − SC@8
  = −0.014 (0/6 sig); Fus@8 − SC@8 = +0.001 (1/6 sig).
=> Once you already have 4–8 full samples, cross-sample *agreement* captures the signal;
per-sample confidence is largely redundant with it. **Probing the samples is a dead end.**
The robust beat-SOTA remains RESULT 7: fuse the SINGLE-CHAIN dynamics (a genuinely
different view) with SC@8 at matched budget (+0.022, 10/25 sig) — the win comes from the
complementary single-generation signal, not from re-probing samples.

## Result 10 — WHY we beat SOTA: we catch the errors self-consistency cannot (KEY)

Self-consistency fails structurally on **high-agreement responses**: when most/all k
samples agree (vote ≥ 0.75), its confidence is ~constant → it cannot distinguish a
confident-correct from a confident-*wrong* consensus. On exactly this subpopulation,
vanilla SC's AUROC ≈ 0.5 (uninformative), yet our answer-confidence dynamics still ranks:

| cell (high-agreement subset, vote≥0.75) | n | acc | SC AUROC | **ours AUROC** |
|-----------------------------------------|----|-----|:--------:|:--------------:|
| gsm8k_qwen8b     | 767 | .975 | ~0.5 | **0.813** |
| math500_qwen8b   | 365 | .962 | ~0.5 | **0.790** |
| olympiad_ministral | 176 | .909 | ~0.5 | **0.694** |
| olympiad_qwen8b  | 347 | .939 | ~0.5 | **0.653** |
| math500_ministral | 275 | .920 | ~0.5 | 0.559 |
| minerva_qwen8b   | 144 | .694 | ~0.5 | 0.487 (small n) |

**This is the mechanism behind Result 7's fusion win.** Self-consistency is blind to
high-confidence errors (agreeing-but-wrong); answer-confidence dynamics detects them.
**Full 31-cell matrix**: on the high-agreement subset (n≥100 cells, 23 of them), our
signal averages **0.659 AUROC** while SC is ~0.5 by construction; ours ≥ 0.65 on 13/23
cells (gsm8k 0.73–0.82, olympiad 0.62–0.75, math500 up to 0.79). This is a clean,
defensible complementarity story: our single-generation signal covers self-consistency's
structural blind spot (confident consensus errors). Script: experiments/mechanism_highconf.py.

## Confound controls (why the signal is real)

- **Within-problem AUROC** (same problem, correct vs wrong traces — controls
  difficulty/gold/domain/wording): cross-sample agreement/stability ranks correct above
  wrong at **0.79**, while per-sample logprob is at chance (**0.52**).
- **Length control:** stability adds +0.059 macro on top of logprob+entropy+**length**;
  within-length-tercile AUROC 0.74 → not a length proxy. (Rules out length, not all
  notions of difficulty; within-problem test is the stronger control.)

---

## Positioning (defensible)

> A **judge-free, verifier-supervised, single-trajectory** uncertainty estimator that
> jointly models **answer-identity dynamics** and **answer-confidence dynamics** along one
> completed reasoning trace. It substantially outperforms all single-generation UQ signals
> and the answer-convergence baseline across 7 model families, with a controlled ablation
> proving the gain comes from **trajectory dynamics, not the endpoint**; and a higher-cost
> hybrid that matches/exceeds 8-sample self-consistency's UQ quality at half the cost on a
> meaningful subset of settings.

**Do NOT claim:** 1× matches SC@8; low-cost allocation beats SC@8 accuracy;
significant on all settings; probing intermediate answers is itself novel (CGR does it).

**Differentiation from prior art:** CGR = single-point certainty for early-stop;
Answer-Convergence = answer identity stabilization; Prefix-Consistency = continuation
reproducibility. **Ours = joint answer-identity + answer-confidence temporal dynamics of
one trace for correctness estimation** (novel, and ablation-verified as non-redundant).

**Beat-SOTA summary (what survived scrutiny):**
1. Single-generation (1×): best low-cost UQ, macro 0.791, > all same-cost SOTA (Result 1).
2. Matched-budget fusion (8×): dynamics ⊕ SC@8 > SC@8, +0.022, sig 10/25 (Result 7).
3. MECHANISM (Result 10): on high-agreement responses SC is at chance (0.5); ours retains
   0.66 macro AUROC — we detect the confident-consensus ERRORS self-consistency cannot.
   This is the principled reason the fusion wins and the paper's core "beat-SOTA" argument.
Dead ends (reported): rich-SC features (R8), confidence-aware per-sample voting (R9),
accuracy-oriented budget allocation. The win is complementarity of the single-chain signal,
not squeezing the sample set.

*Artifacts:* master_table.json, traj_ablation.json, ablation_full.json, hybrid_uq.log,
budget_alloc.log, variance_final.log, figs/mechanism_*.png. All recomputable in seconds
from caches under $EXP_ROOT/{gen,conf,cdyn,labels,sampans,ptrue}.
