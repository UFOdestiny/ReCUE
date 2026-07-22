import json,glob,os,numpy as np
from acd.env import EXP_ROOT, extract_pred_math
from acd import data as dv
from acd import features as D
FO=["agree_frac","last_half_agree","conv_frac","flip_rate","n_distinct","inter_entropy","none_frac","final_stable_run"]
for sub in ["labels","sampans","feats"]:
    (EXP_ROOT/sub).mkdir(exist_ok=True)
for g in sorted(glob.glob(str(EXP_ROOT/"gen"/"*_k8.json"))):
    t=os.path.basename(g)[:-5]
    if t.startswith("smoke"): continue
    lf=EXP_ROOT/"labels"/f"{t}.json"; sf=EXP_ROOT/"sampans"/f"{t}.json"; ff=EXP_ROOT/"feats"/f"{t}.json"
    pf=EXP_ROOT/"probe"/f"{t}_probe.json"
    gen=None
    if not lf.exists() or not sf.exists():
        gen=json.load(open(g))
        if not lf.exists():
            lf.write_text(json.dumps({x["id"]:dv.verify(x,x["primary_text"]) for x in gen})); print("labels",t)
        if not sf.exists():
            sf.write_text(json.dumps({x["id"]:[extract_pred_math(s) for s in x.get("samples",[])] for x in gen})); print("sampans",t)
    if not ff.exists() and pf.exists():
        pr=json.load(open(pf))
        d={r["id"]:{"feat":[D.features(r)[k] for k in FO],"dyn_scalar":D.scalar_confidence(r)} for r in pr}
        ff.write_text(json.dumps(d)); print("feats",t)
print("REFRESH_DONE")
