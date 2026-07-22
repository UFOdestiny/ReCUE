import json,numpy as np
from collections import Counter
from acd.env import EXP_ROOT
from acd import baselines as S
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
def clean(x):
    x=np.asarray(x,float)
    if x.ndim==1:x=x.reshape(-1,1)
    if not np.isfinite(x).all():
        cm=np.nanmin(np.where(np.isfinite(x),x,np.nan),0);cm=np.where(np.isfinite(cm),cm,0.0)
        i=np.where(~np.isfinite(x));x[i]=np.take(cm,i[1])
    return x
def oof(X,y):
    X=clean(X);acc=np.zeros(len(y))
    for s in [2026,7,13]:
        o=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=s).split(X,y):
            c=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(X[tr],y[tr]);o[te]=c.predict_proba(X[te])[:,1]
        acc+=o
    return acc/3
def vote(a,k=8):
    aa=[x for x in a[:k] if x is not None];return Counter(aa).most_common(1)[0][1]/len(aa) if aa else 0
# On the HIGH-agreement subset (SC vote>=0.75), can our signal still rank correct vs wrong?
FO=["agree_frac","last_half_agree","conv_frac","flip_rate","n_distinct","inter_entropy","none_frac","final_stable_run"]
import glob,os; tags=[os.path.basename(f)[:-5] for f in sorted(glob.glob(str(EXP_ROOT/"cdyn"/"*.json"))) if "_s2_" not in f and "_s3_" not in f]
print(f"{'tag':22s}{'n_hi':>6s}{'acc_hi':>7s}{'SC(=const)':>11s}{'ours_hi':>8s}")
for t in tags:
    try:
        cd=json.load(open(EXP_ROOT/"cdyn"/f"{t}.json"));labs=json.load(open(EXP_ROOT/"labels"/f"{t}.json"))
        sa=json.load(open(EXP_ROOT/"sampans"/f"{t}.json"));gen={g["id"]:g for g in json.load(open(EXP_ROOT/"gen"/f"{t}.json"))}
    except: continue
    y=[];F=[];v=[]
    for i in sa:
        if i not in labs or i not in cd: continue
        vf=vote(sa[i]); 
        if vf<0.75: continue  # high-agreement subset
        y.append(labs[i]);F.append(cd[i]["conv"]+cd[i]["cdyn"]+[S.sig_mean_logprob(gen.get(i,{})),S.sig_mean_entropy(gen.get(i,{}))]);v.append(vf)
    y=np.array(y)
    if len(y)<30 or y.sum()==0 or (1-y).sum()==0: 
        print(f"{t:22s} degenerate n={len(y)} pos={int(y.sum()) if len(y) else 0}");continue
    F=clean(np.array(F))
    a_ours=roc_auc_score(y,oof(F,y))
    print(f"{t:22s}{len(y):6d}{y.mean():7.3f}{'~0.5':>11s}{a_ours:8.3f}")
