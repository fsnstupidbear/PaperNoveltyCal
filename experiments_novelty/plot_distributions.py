# plot_distributions.py
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def main(in_csv, outdir, col):
    os.makedirs(outdir, exist_ok=True)
    df=pd.read_csv(in_csv)[[col,"split"]].dropna().copy()
    a=df.loc[df["split"]=="recent", col].values
    b=df.loc[df["split"]=="historical", col].values
    plt.figure(figsize=(6,4))
    plt.hist(a, bins=50, alpha=0.6, density=True, label="Recent")
    plt.hist(b, bins=50, alpha=0.6, density=True, label="Historical")
    plt.legend(); plt.xlabel(col); plt.ylabel("Density")
    plt.title(f"Distribution of {col}: recent vs historical")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"Figure_hist_{col}.png"), dpi=200)
    print("[OK] wrote histogram for", col)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="out_cum/scores_all.csv")
    ap.add_argument("--outdir", default="out_cum/")
    ap.add_argument("--col", default="s_mtn")
    args=ap.parse_args()
    main(args.in_csv, args.outdir, args.col)
