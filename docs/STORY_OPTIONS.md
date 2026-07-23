# ChainUQ — Story / Positioning Options (for you to decide)

The current paper leads with "reading answer-confidence *trajectories*." The new
experiments changed one thing materially: **the temporal ORDER of the trajectory barely
matters** (DUAL − PERM = +0.003, sig 3/31). So a reviewer who runs the permutation test
sinks a "temporal dynamics" claim. Every option below fixes that by reframing the signal
as **multi-prefix answer-commitment** (a distribution over "where/how firmly the model
commits," not a time series). All three are backed by the exact numbers in
`docs/FINAL_RESULTS_FOR_PAPER.md`; they differ in what goes in the *spotlight*.

Common, load-bearing facts all options use:
- 1× tier: ChainUQ 0.791, significantly > best 1× baseline (+0.103, p<0.001, 25/31).
- Matched-8×: fusion 0.898 > SC@8 0.882 (+0.016, p=0.013).
- Mechanism: at unanimous consensus SC-AUROC=0.5, ChainUQ=0.637 (covers SC blind spot).
- Transfer: LODO 0.796 (≈no degradation); single global head 0.767.
- Cost: measured 1.03× a single generation.

---

## OPTION A — "The second axis of compute-aware UQ" (efficiency-first)  ★ recommended

**One-liner.** UQ for reasoning LLMs has spent its budget on *width* (sample more chains,
à la self-consistency). ChainUQ adds an orthogonal *depth* axis — read how the model
commits along one chain — that is (i) the best estimator at 1× cost, (ii) complementary
to sampling so fusing them beats self-consistency at the **same** budget, (iii) real
1.03× wall-clock.

**Spotlight:** cost-tiered main table + the AUROC-vs-latency Pareto + SC blind-spot mechanism.
**Novelty framing:** not "a new feature" but "a new, cheap, complementary *axis*"; the
blind-spot result explains *why* it composes with SC.
**Why it's strong for KDD:** practical impact + honest cost accounting + a beats-SOTA
result that survives scrutiny (matched budget, significant). Sidesteps the order weakness
entirely (order is a minor ablation row, not the thesis).
**Risk:** "incremental vs answer-convergence/DEER" — countered by the +0.103 significant
1× win, the commitment-vs-endpoint ablation (+0.090), and the blind-spot mechanism.
**Title idea:** *ChainUQ: A Depth Axis for Compute-Aware Uncertainty in Reasoning LLMs.*

---

## OPTION B — "Covering self-consistency's blind spot" (mechanism-first)

**One-liner.** Self-consistency is *structurally blind* to confident-consensus errors:
when samples agree, its confidence is constant and its AUROC → 0.5 — yet a large fraction
of those unanimous answers are wrong (244 errors in our data). ChainUQ reads within-trace
commitment and ranks exactly those errors (0.637 where SC=0.5), so fusing the two beats
SC@8 at matched budget.

**Spotlight:** the blind-spot stress test (Table §4) as the centerpiece figure; fusion
table; then the 1× table as "and it's also the best single-pass estimator."
**Novelty framing:** identify + characterize a failure mode of the dominant method (SC),
then provide the covering signal. This is a "diagnosis + fix" paper.
**Why it's strong:** the mechanism is memorable and the claim is precise/defensible;
reviewers like papers that explain *why* SOTA fails.
**Risk:** the beats-SOTA margin is modest (+0.016); mitigate by foregrounding the
blind-spot coverage (large, striking) and framing fusion as the practical payoff.
**Title idea:** *Beyond Agreement: Covering Self-Consistency's Confident-Error Blind Spot.*

---

## OPTION C — "Verifier-supervised commitment estimator that transfers" (generality-first)

**One-liner.** A single verifier-supervised head reading multi-prefix answer-commitment
from one trace is the best 1× UQ, and — unlike per-setting calibrations — it **transfers**:
one global head (no dataset/model id) hits 0.767, leave-one-dataset-out keeps 0.796 with
near-zero degradation, and it extends to non-math MC reasoning.

**Spotlight:** transfer table (LODO/global/LOMO) + non-math generalization + 1× table.
**Novelty framing:** most UQ signals are tuned per model/dataset; ours is a *portable*
commitment signal. Directly answers "isn't this just per-cell feature engineering?"
**Why it's strong:** generality is a clean, testable contribution; label-efficiency (95%
at 25% labels) reinforces "portable + cheap to calibrate."
**Risk:** cross-model transfer is mixed (LOMO), and math→non-math zero-shot fails — must
scope honestly to cross-*dataset* + in-domain non-math. Slightly less punchy than A/B.
**Title idea:** *A Portable Commitment Signal for Judge-Free Reasoning Uncertainty.*

---

## Recommendation

**Option A**, with the blind-spot mechanism (B's centerpiece) as the *why-it-works*
section and transfer (C) as a supporting generality claim. Rationale: A is the framing
where every one of our strongest, most-defensible results is load-bearing and no weak
claim (order, cross-model, non-math-transfer, conformal) is on the critical path. It also
matches the existing title's spirit ("Beyond Self-Consistency") while fixing the
trajectory→commitment naming.

## Section plan under Option A (maps to existing `chap/4.evaluation.tex` RQs)

1. **Main result (RQ1):** cost-tiered table — 1× SOTA (+0.103✓), the depth axis.
2. **Ablation (RQ2):** commitment-not-endpoint (+0.090✓); order-invariance stated
   honestly → justifies "commitment distribution" naming (Block A big table).
3. **Complementarity + beats-SOTA (RQ3):** SC blind spot → matched-8× fusion 0.898>0.882✓.
4. **Generality (RQ4):** LODO 0.796 + global head 0.767 + non-math in-domain +0.076.
5. **Efficiency (RQ5):** 1.03× Pareto. Limitations: no 1×>SC@8; cross-model/ non-math-transfer/ conformal-shift negatives stated.

## Naming decision needed from you
- Keep method name **ChainUQ** (yes/no).
- Reframe **"temporal trajectory" → "multi-prefix commitment"** throughout (recommended yes).
- Keep component names **PrefixProbe / DualTrace / ConsensusFusion** (they still fit).
