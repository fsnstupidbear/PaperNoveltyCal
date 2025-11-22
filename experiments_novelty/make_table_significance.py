# make_table_significance.py
import os, argparse, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return (a.mean()-b.mean()) / (s + 1e-12)

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    # O(N log N) 近似
    A = np.sort(a); B = np.sort(b)
    i=j=0; gt=lt=0
    while i<len(A) and j<len(B):
        if A[i] > B[j]:
            gt += len(A)-i
            j += 1
        elif A[i] < B[j]:
            lt += len(B)-j
            i += 1
        else:
            ai=i
            while i<len(A) and A[i]==A[ai]: i+=1
            bj=j
            while j<len(B) and B[j]==B[bj]: j+=1
            # ties 不计入 gt/lt
    n = len(a)*len(b)
    return (gt - lt) / (n + 1e-12)

def eval_one(df, col):
    df = df[[col,"split"]].dropna().copy()
    y = (df["split"]=="recent").astype(int).values
    s = df[col].values
    a = df.loc[df["split"]=="recent", col].values
    b = df.loc[df["split"]=="historical", col].values
    # Welch t
    t, p_two = stats.ttest_ind(a, b, equal_var=False, alternative="greater")  # recent > historical
    # Mann-Whitney U (recent > historical)
    U, p_u = stats.mannwhitneyu(a, b, alternative="greater")
    # AUC / PR-AUC
    auc = roc_auc_score(y, s)
    prauc = average_precision_score(y, s)
    # 简单阈值：最大化 F1
    thr_grid = np.quantile(s, np.linspace(0.01,0.99,99))
    best = (None, -1)
    for t0 in thr_grid:
        yp = (s>=t0).astype(int)
        f1 = f1_score(y, yp)
        if f1>best[1]: best=(t0,f1)
    thr = best[0]
    yp = (s>=thr).astype(int)
    prec = precision_score(y, yp, zero_division=0)
    rec  = recall_score(y, yp, zero_division=0)
    f1   = f1_score(y, yp, zero_division=0)
    acc  = (yp==y).mean()
    return {
        "score_col": col,
        "N_total": len(df),
        "N_recent": int((df["split"]=="recent").sum()),
        "N_historical": int((df["split"]=="historical").sum()),
        "recent_mean": float(a.mean()), "historical_mean": float(b.mean()),
        "mean_diff": float(a.mean()-b.mean()),
        "Welch_t": float(t), "Welch_p(one-sided)": float(p_two),
        "MannWhitney_U": float(U), "MannWhitney_p(one-sided)": float(p_u),
        "Cohen_d": float(cohen_d(a,b)), "Cliffs_delta": float(cliffs_delta(a,b)),
        "ROC_AUC": float(auc), "PR_AUC": float(prauc),
        "best_thr": float(thr), "Prec": float(prec), "Rec": float(rec), "F1": float(f1), "Acc": float(acc),
    }

def main(in_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(in_csv)
    need = ["s_m","s_t","s_mtn","S"]
    rows = [eval_one(df, c) for c in need if c in df.columns]
    tab = pd.DataFrame(rows)
    tab = tab.sort_values("ROC_AUC", ascending=False)
    tab.to_csv(os.path.join(outdir,"Table_significance.csv"), index=False)
    # 生成 md
    md = "|Score|N|AUC|PR-AUC|Welch p(one-sided)|Cohen d|Cliff δ|Acc|F1|\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        md += f"|{r['score_col']}|{r['N_total']}|{r['ROC_AUC']:.3f}|{r['PR_AUC']:.3f}|{r['Welch_p(one-sided)']:.4f}|{r['Cohen_d']:.3f}|{r['Cliffs_delta']:.3f}|{r['Acc']:.3f}|{r['F1']:.3f}|\n"
    with open(os.path.join(outdir,"Table_significance.md"),"w",encoding="utf-8") as f:
        f.write(md)
    print("[OK] wrote Table_significance.{csv,md} in", outdir)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="out_cum/scores_all.csv")
    ap.add_argument("--outdir", default="out_cum/")
    args = ap.parse_args()
    main(args.in_csv, args.outdir)
