import os, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def main(outdir, scores_csv, score_col="s_mtn"):
    pool = []
    with open(os.path.join(outdir, "pool_ALL.jsonl"), "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip(): pool.append(json.loads(ln))
    df = pd.DataFrame(pool)[["id","split"]]
    sco = pd.read_csv(scores_csv)
    if score_col not in sco.columns:
        raise KeyError(f"column {score_col} not in {scores_csv}")
    df = df.merge(sco[["id",score_col]], on="id", how="left")
    df = df.dropna(subset=[score_col, "split"])
    y = (df["split"]=="recent").astype(int).values
    s = df[score_col].values
    auc = roc_auc_score(y, s)
    fpr,tpr,_ = roc_curve(y, s)
    print({"AUC": float(auc), "N": int(len(y)), "score_col": score_col})
    plt.figure(figsize=(4.8,4.0), dpi=200)
    plt.plot(fpr,tpr,label=f"{score_col} (AUC={auc:.3f})"); plt.plot([0,1],[0,1],'--',label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"Global discrimination: {score_col}")
    plt.legend(); plt.tight_layout()
    figp = os.path.join(outdir, f"Figure_global_{score_col}.png")
    plt.savefig(figp); plt.close()
    print("[OK] wrote", figp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--score_col", default="s_mtn")  # s_mn, s_tn 也可测
    args = ap.parse_args()
    main(args.outdir, args.scores_csv, args.score_col)
