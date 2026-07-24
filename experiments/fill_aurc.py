"""Compute missing AURC numbers, endpoint-only, 29 headline cells.
(1) Table 2: mean_logprob + deepconf per-dataset AURC, macro, excess, risk@10/20/50.
(2) Table 3 ablation blocks (b) and (d): macro AURC for each representation.
"""
import json, numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from recue.env import EXP_ROOT
from recue import baselines as S
from recue.metrics import aurc, risk_at_coverage

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


Z1 = np.load(f"{EXP_ROOT}/ladder_feats.npz"); Z2 = np.load(f"{EXP_ROOT}/aime_feats.npz")
cells = {}
for Z in [Z1, Z2]:
    for t in sorted({k.split("::")[0] for k in Z.files}):
        ds = t.split("_")[0]
        if ds == "amc23" or t in EXCLUDE or ds not in HEAD:
            continue
        cells[t] = Z

# ---- assemble per-cell arrays ----
D = {}
for t, Z in cells.items():
    y = Z[f"{t}::y"]
    ARC = np.hstack([Z[f"{t}::AGREE"], Z[f"{t}::FLL"], Z[f"{t}::FCONF"]])
    TUP = Z[f"{t}::TRACE"]; AGREE = Z[f"{t}::AGREE"]
    pt = Z[f"{t}::PT"].ravel()
    gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{t}.json").read_text())}
    conf = json.loads((EXP_ROOT / "conf" / f"{t}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{t}.json").read_text())
    probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{t}_probe.json").read_text())}
    reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{t}_reb.json").read_text())}
    tf_path = EXP_ROOT / "tforce" / f"{t}_tf.json"
    tf = json.loads(tf_path.read_text()) if tf_path.exists() else {}
    mlp, dc, supp = [], [], []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
            continue
        mlp.append(S.sig_mean_logprob(gen[rid])); dc.append(S.sig_deepconf_bottom(gen[rid]))
        rec = tf.get(rid)
        if rec is None:
            supp.append([-10.0, -10.0, -10.0])
        else:
            tl = rec["tok_lps"]; supp.append([float(np.mean(tl)), float(np.min(tl)), float(rec["first_lp"])])
    D[t] = {"y": y, "ARC": ARC, "TUP": TUP, "AGREE": AGREE, "pt": pt,
            "mlp": np.array(mlp), "dc": np.array(dc), "supp": np.array(supp),
            "ds": t.split("_")[0]}

# =================== Table 2: mean_logprob + deepconf =====================
print("=== Table 2 missing rows (scalar baselines) ===")
def scalar_row(key):
    bd = defaultdict(list); mac, ex, r10, r20, r50 = [], [], [], [], []
    for t, d in D.items():
        s = d[key]; y = d["y"]; rc = aurc(s, y); acc = y.mean()
        bd[d["ds"]].append(rc); mac.append(rc)
        if acc < 1: ex.append(rc / (1 - acc))
        r10.append(risk_at_coverage(s, y, .10)); r20.append(risk_at_coverage(s, y, .20)); r50.append(risk_at_coverage(s, y, .50))
    return ({dd: float(np.mean(bd[dd])) for dd in HEAD if bd[dd]},
            float(np.mean(mac)), float(np.mean(ex)),
            float(np.mean(r10)), float(np.mean(r20)), float(np.mean(r50)))

out = {}
for key, name in [("mlp", "Mean log-probability"), ("dc", "DeepConf-bottom")]:
    bd, mac, ex, r10, r20, r50 = scalar_row(key)
    cs = "  ".join(f"{d[:5]}={bd.get(d, float('nan')):.3f}" for d in HEAD)
    print(f"{name:22s} {cs}  macro={mac:.3f} excess={ex:.3f} r10={r10:.3f} r20={r20:.3f} r50={r50:.3f}")
    out[name] = {"by_ds": bd, "macro": mac, "excess": ex, "r10": r10, "r20": r20, "r50": r50}

# =================== Table 3 blocks (b) and (d) AURC ======================
print("\n=== Table 3 block (b)/(d) AURC (macro) ===")
def macro_aurc(featfn):
    vals = []
    for t, d in D.items():
        s = oof(featfn(d), d["y"]); vals.append(aurc(s, d["y"]))
    return float(np.mean(vals))

# block (b): P(True) is a raw scalar (not learned) -> its AURC uses the scalar directly
def macro_aurc_scalar(key):
    return float(np.mean([aurc(D[t][key], D[t]["y"]) for t in D if np.isfinite(D[t][key]).all()]))

blocks = {
    "P(True) [scalar]": lambda: macro_aurc_scalar("pt"),
    "agreement bit": lambda: macro_aurc(lambda d: d["AGREE"]),
    "P(True)+agree bit": lambda: macro_aurc(lambda d: np.hstack([d["pt"].reshape(-1, 1), d["AGREE"]])),
    "P(True)+ARC": lambda: macro_aurc(lambda d: np.hstack([d["pt"].reshape(-1, 1), d["ARC"]])),
    "support": lambda: macro_aurc(lambda d: d["supp"]),
    "TUP+support": lambda: macro_aurc(lambda d: np.hstack([d["TUP"], d["supp"]])),
    "ARC+support": lambda: macro_aurc(lambda d: np.hstack([d["ARC"], d["supp"]])),
    "ReCUE+support": lambda: macro_aurc(lambda d: np.hstack([d["ARC"], d["TUP"], d["supp"]])),
}
for name, fn in blocks.items():
    v = fn(); out[name] = v
    print(f"{name:22s} AURC={v:.3f}")

json.dump(out, open(f"{EXP_ROOT}/fill_aurc.json", "w"), indent=2)
print("\nsaved fill_aurc.json")
