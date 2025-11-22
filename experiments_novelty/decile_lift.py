# decile_lift.py
import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

def wilson(p, n, z=1.96):
    if n==0: return (0,0)
    denom = 1+z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*np.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom
    return center-half, center+half

def main(in_csv, outdir, col):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(in_csv)[[col,"split"]].dropna().copy()
    df["y"] = (df["split"]=="recent").astype(int)
    df = df.sort_values(col)
    df["bin"] = pd.qcut(df[col], 10, labels=False, duplicates="drop")
    base = df["y"].mean()
    rows=[]
    for b, g in df.groupby("bin"):
        n=len(g); pos=g["y"].sum(); p=pos/max(n,1)
        lo, hi = wilson(p, n)
        rows.append({"bin":int(b), "n":n, "p":p, "lift":p/(base+1e-12), "lo":lo/(base+1e-12), "hi":hi/(base+1e-12)})
    tab=pd.DataFrame(rows).sort_values("bin")
    tab.to_csv(os.path.join(outdir,f"decile_lift_{col}.csv"), index=False)

    plt.figure(figsize=(7,4))
    plt.errorbar(range(len(tab)), tab["lift"],
                 yerr=[tab["lift"]-tab["lo"], tab["hi"]-tab["lift"]],
                 fmt='o-', capsize=3)
    plt.axhline(1.0, ls='--')
    plt.xticks(range(len(tab)), [f"D{b+1}" for b in tab["bin"]])
    plt.ylabel("Lift = Recent rate / base rate")
    plt.title(f"Decile lift by {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"Figure_decile_lift_{col}.png"), dpi=200)
    print("[OK] wrote decile lift figure & csv for", col)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="out_cum/scores_all.csv")
    ap.add_argument("--outdir", default="out_cum/")
    ap.add_argument("--col", default="s_mtn")
    args=ap.parse_args()
    main(args.in_csv, args.outdir, args.col)
