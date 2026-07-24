"""No-GPU rebuttal experiments (cached features only), endpoint-only, 29 headline cells.
Exp4: strong supervised SC@8 fusion baseline vs ReCUE+SC@8.
Exp5: full ReCUE (ARC+TUP) transfer (LODO / global / LOMO) beside ARC and TUP.
Exp2A: answer-leakage natural grouping (reasoning body contains final answer or not).
Writes rebuttal_nogpu.json.
"""
import json, re, numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from recue.env import EXP_ROOT
from recue import baselines as S

SEEDS = [2026, 7, 13, 42, 100]
EXCLUDE = {"aime_qwen35_9b_k8", "math500_llama8b_k8"}
HEAD = ["gsm8k", "math500", "minerva", "olympiad", "aime"]


def clean(x):
    x = np.asarray(x, float).reshape(len(x), -1)
    if not np.isfinite(x).all():
        cm = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
        i = np.where(~np.isfinite(x)); x[i] = np.take(cm, i[1])
    return x


def oof(X, y):
    X = clean(X); a = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        a += o
    return a / len(SEEDS)


def fit_pred(Xtr, ytr, Xte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(clean(Xtr), ytr)
    return m.predict_proba(clean(Xte))[:, 1]


def _num_equiv_present(text, ans):
    """crude: is the normalized answer string present in the reasoning text?"""
    if not ans:
        return False
    a = ans.strip()
    if a and a in text:
        return True
    # numeric equivalence: try float match of any number token
    try:
        av = float(re.sub(r"[^0-9.\-]", "", a))
        for m in re.findall(r"-?\d+\.?\d*", text):
            if abs(float(m) - av) < 1e-6:
                return True
    except Exception:
        pass
    return False


Z1 = np.load(f"{EXP_ROOT}/ladder_feats.npz"); Z2 = np.load(f"{EXP_ROOT}/aime_feats.npz")
cells = {}
for Z in [Z1, Z2]:
    for t in sorted({k.split("::")[0] for k in Z.files}):
        ds = t.split("_")[0]
        if ds == "amc23" or t in EXCLUDE or ds not in HEAD:
            continue
        cells[t] = Z

# ---- assemble per-cell features ----
data = {}   # tag -> dict
for t, Z in cells.items():
    y = Z[f"{t}::y"]
    ARC = np.hstack([Z[f"{t}::AGREE"], Z[f"{t}::FLL"], Z[f"{t}::FCONF"]])
    TUP = Z[f"{t}::TRACE"]
    gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{t}.json").read_text())}
    conf = json.loads((EXP_ROOT / "conf" / f"{t}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{t}.json").read_text())
    probe = {r["id"]: r for r in json.loads((EXP_ROOT / "probe" / f"{t}_probe.json").read_text())}
    reb = {r["id"]: r for r in json.loads((EXP_ROOT / "rebuttal" / f"{t}_reb.json").read_text())}
    saf = EXP_ROOT / "sampans" / f"{t}.json"; sa = json.loads(saf.read_text()) if saf.exists() else {}
    vote, ent, margin, mlp, minlp, ansll, leak = [], [], [], [], [], [], []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
            continue
        ans = [a for a in sa.get(rid, [])[:8] if a is not None]; c = Counter(ans); tot = sum(c.values())
        top = c.most_common(2)
        vf = top[0][1] / tot if tot else 0.0
        vm = (top[0][1] - (top[1][1] if len(top) > 1 else 0)) / tot if tot else 0.0
        he = -sum((v / tot) * np.log(v / tot) for v in c.values()) if tot else 0.0
        vote.append(vf); margin.append(vm); ent.append(he)
        mlp.append(S.sig_mean_logprob(gen[rid])); minlp.append(S.sig_min_logprob(gen[rid]))
        full = reb[rid].get("full") or {}; tl = full.get("tok_lps") or []
        ansll.append(float(np.mean(tl)) if tl else -10.0)
        # leakage: does reasoning body contain the final answer?
        txt = gen[rid].get("primary_text", "")
        fa = probe[rid].get("final_answer")
        leak.append(1 if _num_equiv_present(txt, fa) else 0)
    data[t] = {"y": y, "ARC": ARC, "TUP": TUP,
               "vote": np.array(vote).reshape(-1, 1), "ent": np.array(ent).reshape(-1, 1),
               "margin": np.array(margin).reshape(-1, 1), "mlp": np.array(mlp).reshape(-1, 1),
               "minlp": np.array(minlp).reshape(-1, 1), "ansll": np.array(ansll).reshape(-1, 1),
               "leak": np.array(leak), "ds": t.split("_")[0], "model": t.split("_", 1)[1]}

# =========================================================================
# Exp4: strong supervised SC@8 fusion
# =========================================================================
print("=== Exp4: strong supervised SC@8 ===")
def macro(scorefn):
    vals = []
    for t, d in data.items():
        s = scorefn(d)
        vals.append(roc_auc_score(d["y"], s))
    return float(np.mean(vals))

sc8_vote = macro(lambda d: d["vote"].ravel())
strong_sc = macro(lambda d: oof(np.hstack([d["vote"], d["ent"], d["margin"], d["mlp"], d["minlp"], d["TUP"], d["ansll"]]), d["y"]))
sc8_tup = macro(lambda d: oof(np.hstack([d["vote"], d["ent"], d["TUP"]]), d["y"]))
recue_sc8 = macro(lambda d: oof(np.hstack([d["ARC"], d["TUP"], d["vote"], d["ent"]]), d["y"]))
print(f"  SC@8 vote fraction           {sc8_vote:.3f}")
print(f"  Strong supervised SC@8       {strong_sc:.3f}")
print(f"  SC@8 + pooled TUP            {sc8_tup:.3f}")
print(f"  ReCUE + SC@8                 {recue_sc8:.3f}")
exp4 = {"sc8_vote": sc8_vote, "strong_supervised_sc8": strong_sc,
        "sc8_plus_tup": sc8_tup, "recue_sc8": recue_sc8}

# =========================================================================
# Exp5: full ReCUE transfer (LODO / global / LOMO), beside ARC and TUP
# =========================================================================
print("\n=== Exp5: full ReCUE transfer ===")
FS = {"ARC": lambda d: d["ARC"], "TUP": lambda d: d["TUP"],
      "ReCUE": lambda d: np.hstack([d["ARC"], d["TUP"]])}
models = sorted({d["model"] for d in data.values()})
exp5 = {}
for name, fs in FS.items():
    # LODO
    tr_by_tgt, id_by_tgt = defaultdict(list), defaultdict(list)
    for tgt in HEAD:
        for m in models:
            tc = next((d for d in data.values() if d["ds"] == tgt and d["model"] == m), None)
            if tc is None: continue
            src = [d for d in data.values() if d["model"] == m and d["ds"] != tgt]
            if len({d["ds"] for d in src}) < 3: continue
            Xtr = np.vstack([fs(d) for d in src]); ytr = np.concatenate([d["y"] for d in src])
            y = tc["y"]
            if y.sum() == 0 or (1 - y).sum() == 0: continue
            tr_by_tgt[tgt].append(roc_auc_score(y, fit_pred(Xtr, ytr, fs(tc))))
            id_by_tgt[tgt].append(roc_auc_score(y, oof(fs(tc), y)))
    lodo_t = np.mean([np.mean(tr_by_tgt[k]) for k in tr_by_tgt])
    lodo_i = np.mean([np.mean(id_by_tgt[k]) for k in id_by_tgt])
    # global head leave-one-cell-out
    tags = sorted(data.keys()); loco = []
    for t in tags:
        Xtr = np.vstack([fs(data[o]) for o in tags if o != t]); ytr = np.concatenate([data[o]["y"] for o in tags if o != t])
        y = data[t]["y"]
        if y.sum() == 0 or (1 - y).sum() == 0: continue
        loco.append(roc_auc_score(y, fit_pred(Xtr, ytr, fs(data[t]))))
    # LOMO
    lomo = []
    for m in models:
        src = [d for d in data.values() if d["model"] != m]; tgt = [d for d in data.values() if d["model"] == m]
        if not tgt: continue
        Xtr = np.vstack([fs(d) for d in src]); ytr = np.concatenate([d["y"] for d in src])
        for tc in tgt:
            y = tc["y"]
            if y.sum() == 0 or (1 - y).sum() == 0: continue
            lomo.append(roc_auc_score(y, fit_pred(Xtr, ytr, fs(tc))))
    exp5[name] = {"lodo_transfer": float(lodo_t), "lodo_indomain": float(lodo_i),
                  "global_loco": float(np.mean(loco)), "lomo": float(np.mean(lomo))}
    print(f"  {name:6s} LODO {lodo_t:.3f} (in-dom {lodo_i:.3f})  global {np.mean(loco):.3f}  LOMO {np.mean(lomo):.3f}")

# =========================================================================
# Exp2A: answer-leakage natural grouping
# =========================================================================
print("\n=== Exp2A: answer leakage grouping ===")
def macro_subset(feat, mask_fn):
    va, vr = [], []; na = []
    for t, d in data.items():
        m = mask_fn(d); y = d["y"]
        for lab, sel in [("with", m == 1), ("without", m == 0)]:
            if sel.sum() < 30 or y[sel].sum() == 0 or (1 - y[sel]).sum() == 0:
                continue
    return None

exp2a = {}
for name, fs in [("ARC", lambda d: d["ARC"]), ("ReCUE", lambda d: np.hstack([d["ARC"], d["TUP"]]))]:
    for grp, want in [("with_answer", 1), ("without_answer", 0)]:
        vals = []; frac = []
        for t, d in data.items():
            sel = d["leak"] == want; y = d["y"]
            if sel.sum() < 30 or y[sel].sum() == 0 or (1 - y[sel]).sum() == 0:
                continue
            s = oof(fs(d)[sel], y[sel])
            vals.append(roc_auc_score(y[sel], s)); frac.append(sel.mean())
        exp2a[f"{name}_{grp}"] = {"auroc": float(np.mean(vals)), "n_cells": len(vals),
                                  "mean_frac": float(np.mean(frac)) if frac else 0.0}
        print(f"  {name:6s} {grp:16s} AUROC {np.mean(vals):.3f}  ({len(vals)} cells, frac {np.mean(frac):.2f})")

# overall leakage rate
allleak = np.concatenate([d["leak"] for d in data.values()])
print(f"  overall leakage rate: {allleak.mean():.3f}")
exp2a["overall_leak_rate"] = float(allleak.mean())

json.dump({"exp4": exp4, "exp5": exp5, "exp2a": exp2a},
          open(f"{EXP_ROOT}/rebuttal_nogpu.json", "w"), indent=2)
print("\nsaved rebuttal_nogpu.json")
