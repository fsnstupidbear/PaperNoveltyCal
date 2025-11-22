# eval_binary_plus.py  — Global binary significance + tail/decile enrichment for novelty scores
# Input: CSV with columns: ['id','split', <score columns e.g., s_m, s_t, s_mtn, S ...>]
# Usage example:
#   python eval_binary_plus.py --scores_csv out_cum/scores_all.csv --score_col s_mtn --outdir out_cum/ --seed 42 --boot 2000

import argparse, os, json, math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.utils import resample
from scipy import stats
import matplotlib.pyplot as plt

def bootstrap_ci_auc(y, s, rounds=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)

def topk_precision(y, s, ks):
    # ks are percentages (e.g., [1,5,10,20]) of positives in top portion by score
    order = np.argsort(-s)
    y_sorted = y[order]
    res = {}
    for k in ks:
        m = max(1, int(round(len(y) * (k/100.0))))
        p = y_sorted[:m].mean()
        res[k] = float(p)
    return res

def tpr_at_fpr(y_true, y_score, fpr_targets=(0.01, 0.05, 0.10)):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    out = {}
    for tgt in fpr_targets:
        key = f"{tgt:.2f}"          # 统一成 '0.01' / '0.05' / '0.10'
        mask = fpr <= (tgt + 1e-12) # 容忍微小数值误差
        out[key] = float(np.max(tpr[mask])) if mask.any() else 0.0
    return out


def plot_hist_kde(y, s, out_png, title):
    # two simple overlaid histograms + KDE-like via density=True
    pos = s[y==1]; neg = s[y==0]
    plt.figure(figsize=(6,4))
    bins = 50
    plt.hist(neg, bins=bins, alpha=0.5, density=True, label="Historical (label=0)")
    plt.hist(pos, bins=bins, alpha=0.5, density=True, label="Recent (label=1)")
    try:
        # crude KDE via gaussian_kde if available
        kde_neg = stats.gaussian_kde(neg)
        kde_pos = stats.gaussian_kde(pos)
        xs = np.linspace(min(s), max(s), 300)
        plt.plot(xs, kde_neg(xs), label="Neg KDE")
        plt.plot(xs, kde_pos(xs), label="Pos KDE")
    except Exception:
        pass
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_roc(y, s, out_png, title, auc_ci=None):
    fpr, tpr, _ = roc_curve(y, s)
    A = roc_auc_score(y, s)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={A:.3f}" + (f" (95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f})" if auc_ci else ""))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_pr(y, s, out_png, title):
    precision, recall, _ = precision_recall_curve(y, s)
    A = auc(recall, precision)
    plt.figure(figsize=(5,5))
    plt.plot(recall, precision, label=f"PR-AUC={A:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_lift_curve(base_rate, prec_at_k, out_png, title):
    # prec_at_k: dict {k%: precision}
    ks = sorted(prec_at_k.keys())
    lifts = [prec_at_k[k]/base_rate if base_rate>0 else 0.0 for k in ks]
    plt.figure(figsize=(6,4))
    plt.plot(ks, lifts, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Top-k % of ranked papers by score")
    plt.ylabel("Lift over base rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main(scores_csv, score_col, outdir, seed, boot_rounds):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(scores_csv)
    if "split" not in df.columns:
        raise KeyError("CSV must contain column 'split' with values {'recent','historical'}.")

    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not found. Available: {list(df.columns)}")

    # label: recent=1, historical=0
    y = (df["split"].astype(str)=="recent").astype(int).values
    s = pd.to_numeric(df[score_col], errors="coerce").values

    # drop NaN
    mask = ~np.isnan(s)
    kept = mask.sum()
    dropped = len(s) - kept
    y = y[mask]; s = s[mask]
    kept_recent = int((y==1).sum()); kept_hist = int((y==0).sum())

    # Welch t-test
    s_pos = s[y==1]; s_neg = s[y==0]
    t_stat, p_t = stats.ttest_ind(s_pos, s_neg, equal_var=False, nan_policy="omit")

    # Mann-Whitney U (equiv to AUC)
    u_stat, p_u = stats.mannwhitneyu(s_pos, s_neg, alternative="two-sided")
    auc_mw = u_stat / (len(s_pos)*len(s_neg))

    # Cohen's d
    m1, m0 = np.mean(s_pos), np.mean(s_neg)
    v1, v0 = np.var(s_pos, ddof=1), np.var(s_neg, ddof=1)
    # pooled SD (Hedges' g could also be reported)
    sp = math.sqrt(((len(s_pos)-1)*v1 + (len(s_neg)-1)*v0) / (len(s_pos)+len(s_neg)-2))
    d = (m1 - m0) / (sp if sp>0 else 1.0)

    # ROC/PR + bootstrap CI for AUC
    rocAUC = roc_auc_score(y, s)
    lo, hi = bootstrap_ci_auc(y, s, rounds=boot_rounds, seed=seed)
    precision, recall, _ = precision_recall_curve(y, s)
    prAUC = auc(recall, precision)

    # TPR at small FPR
    tpr_small = tpr_at_fpr(y, s, fpr_targets=(0.01, 0.05, 0.10))

    # Top-k precision & Lift
    base_rate = float((y==1).mean())
    ks = [1,5,10,20]
    prec_at_k = topk_precision(y, s, ks)
    lift_at_k = {k: (prec_at_k[k]/base_rate if base_rate>0 else 0.0) for k in ks}

    # Plots
    plot_hist_kde(y, s, os.path.join(outdir, f"Fig_dist_{score_col}.png"),
                  f"Distribution of {score_col} (recent vs historical)")
    plot_roc(y, s, os.path.join(outdir, f"Fig_ROC_{score_col}.png"),
             f"ROC of {score_col}", auc_ci=(lo,hi))
    plot_pr(y, s, os.path.join(outdir, f"Fig_PR_{score_col}.png"),
            f"PR of {score_col}")
    plot_lift_curve(base_rate, prec_at_k,
                    os.path.join(outdir, f"Fig_Lift_{score_col}.png"),
                    f"Lift@k of {score_col} (base={base_rate:.3f})")

    summary = {
        "dataset": scores_csv,
        "score_col": score_col,
        "class_balance": {"recent_rate": base_rate, "recent": int((y==1).sum()), "historical": int((y==0).sum())},
        "row_counts": {"total": int(len(df)), "kept": int(kept), "dropped": int(dropped),
                       "kept_recent": kept_recent, "kept_historical": kept_hist},
        "welch_t": {"t": float(t_stat), "p": float(p_t), "mean_recent": float(m1), "mean_historical": float(m0)},
        "mannwhitney_u": {"U": float(u_stat), "p": float(p_u), "AUC_equiv": float(auc_mw)},
        "effect_size": {"cohens_d": float(d)},
        "roc_auc": {"AUC": float(rocAUC), "CI95": [float(lo), float(hi)]},
        "pr_auc": float(prAUC),
        "TPR_at_FPR": tpr_small,
        "prec_at_k": prec_at_k,
        "lift_at_k": lift_at_k
    }
    with open(os.path.join(outdir, f"binary_{score_col}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] wrote", os.path.join(outdir, f"binary_{score_col}_summary.json"))

    # Also tabular CSV for paper
    rows = []
    rows.append(["Welch t-test p", p_t])
    rows.append(["Mann-Whitney U p", p_u])
    rows.append(["AUC (ROC)", rocAUC])
    rows.append(["AUC 95% CI lo", lo])
    rows.append(["AUC 95% CI hi", hi])
    rows.append(["PR-AUC", prAUC])
    rows.append(["Cohen's d", d])
    rows.append(["TPR@FPR=1%", tpr_small["0.01"]])
    rows.append(["TPR@FPR=5%", tpr_small["0.05"]])
    rows.append(["TPR@FPR=10%", tpr_small["0.10"]])
    for k in ks:
        rows.append([f"Precision@{k}%", prec_at_k[k]])
        rows.append([f"Lift@{k}%", lift_at_k[k]])
    tab = pd.DataFrame(rows, columns=["Metric","Value"])
    tab.to_csv(os.path.join(outdir, f"Table_binary_{score_col}.csv"), index=False)
    with open(os.path.join(outdir, f"Table_binary_{score_col}.md"), "w", encoding="utf-8") as f:
        f.write("| Metric | Value |\n|---|---|\n")
        for m,v in rows:
            f.write(f"| {m} | {v:.6f} |\n")
    print("[OK] wrote Table & Figures to", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--score_col", default="s_mtn")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    main(args.scores_csv, args.score_col, args.outdir, args.seed, args.boot)
