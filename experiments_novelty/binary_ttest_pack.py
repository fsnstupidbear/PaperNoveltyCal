# binary_ttest_pack.py
# -*- coding: utf-8 -*-
import argparse, os, json, math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def cohen_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = ((nx-1)*sx + (ny-1)*sy) / (nx+ny-2) if nx+ny-2>0 else 0.0
    sp = math.sqrt(max(sp, 1e-12))
    return (np.mean(x) - np.mean(y)) / sp

def hedges_g(x, y):
    d = cohen_d(x, y)
    n = len(x) + len(y)
    J = 1.0 - 3.0/(4.0*n - 9.0) if n>3 else 1.0
    return d * J

def cliffs_delta_from_mwu(u_stat, n1, n2):
    # delta = 2*U/(n1*n2) - 1
    return 2.0 * (u_stat / (n1*n2)) - 1.0

def best_threshold_acc(y_true, scores):
    # 返回使分类准确率最高的阈值与各指标（按正类=Recent）
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores)
    # 用 PR 曲线的阈值集合做扫描
    try:
        prec, rec, thr = precision_recall_curve(y, s)
        # precision_recall_curve 会返回 len(thr)+1 个点，构造阈值列表
        cand = np.r_[ -np.inf, thr, np.inf ]
    except Exception:
        # fallback：用分数的唯一值
        cand = np.r_[ -np.inf, np.unique(s), np.inf ]
    best = (-1, None, None, None, None)  # acc, thr, f1, prec, rec
    pos = (y==1).sum()
    neg = (y==0).sum()
    for t in cand:
        yhat = (s >= t).astype(int)
        acc = (yhat==y).mean()
        tp = ((yhat==1)&(y==1)).sum()
        fp = ((yhat==1)&(y==0)).sum()
        fn = ((yhat==0)&(y==1)).sum()
        precision = tp/(tp+fp+1e-12)
        recall    = tp/(tp+fn+1e-12)
        f1 = 2*precision*recall/(precision+recall+1e-12)
        if acc > best[0]:
            best = (acc, t, f1, precision, recall)
    return {"best_thr": float(best[1]),
            "Acc": float(best[0]),
            "F1": float(best[2]),
            "Prec": float(best[3]),
            "Rec": float(best[4])}

def run_one(df, col, outdir, label_col="split", pos_label="recent",
            by_field=False, balanced=False, boot=0, seed=42):
    rng = np.random.default_rng(seed)

    # 取 label
    if label_col not in df.columns:
        raise RuntimeError(f"Missing label column '{label_col}' in CSV.")
    y = (df[label_col].astype(str).str.lower()==pos_label).astype(int)

    # 分数列
    if col not in df.columns:
        raise RuntimeError(f"Missing score column '{col}' in CSV.")
    s = pd.to_numeric(df[col], errors="coerce")
    keep = s.notna() & y.notna()
    df = df.loc[keep].copy()
    s = s.loc[keep].to_numpy()
    y = y.loc[keep].to_numpy()

    # 可选：平衡抽样（仅用于统计/作图，不改原 CSV）
    if balanced:
        pos_idx = np.where(y==1)[0]
        neg_idx = np.where(y==0)[0]
        k = min(len(pos_idx), len(neg_idx))
        pos_sel = rng.choice(pos_idx, k, replace=False)
        neg_sel = rng.choice(neg_idx, k, replace=False)
        sel = np.r_[pos_sel, neg_sel]
        s, y = s[sel], y[sel]

    # 基础 AUC
    try:
        roc_auc = roc_auc_score(y, s)
        pr_auc  = average_precision_score(y, s)
    except Exception:
        roc_auc, pr_auc = np.nan, np.nan

    # 统计检验（Recent vs Historical）
    x = s[y==1]
    z = s[y==0]
    # Welch t（稳妥）
    t_stat, t_p = stats.ttest_ind(x, z, equal_var=False, alternative="greater")
    # 也给出双侧 p 以备写作
    t_stat2, t_p_two = stats.ttest_ind(x, z, equal_var=False, alternative="two-sided")
    # Mann–Whitney U（非参数）
    try:
        u_stat, u_p = stats.mannwhitneyu(x, z, alternative="greater")
    except ValueError:
        u_stat, u_p = np.nan, np.nan
    # 效应量
    d = cohen_d(x, z)
    g = hedges_g(x, z)
    delta = cliffs_delta_from_mwu(u_stat, len(x), len(z)) if np.isfinite(u_stat) else np.nan

    # 最优阈值扫描
    thr_pack = best_threshold_acc(y, s)

    # Bootstrap（可选）
    boot_ci = {}
    if boot and boot>0:
        diffs = []
        for _ in range(int(boot)):
            bx = rng.choice(x, size=len(x), replace=True)
            bz = rng.choice(z, size=len(z), replace=True)
            diffs.append(np.mean(bx) - np.mean(bz))
        diffs = np.sort(diffs)
        lo = float(np.percentile(diffs, 2.5))
        hi = float(np.percentile(diffs, 97.5))
        boot_ci = {"mean_diff_CI": [lo, hi]}

    # 输出一行
    row = {
        "score_col": col,
        "N_total": int(len(s)),
        "N_recent": int((y==1).sum()),
        "N_historical": int((y==0).sum()),
        "recent_mean": float(np.mean(x)) if len(x) else np.nan,
        "historical_mean": float(np.mean(z)) if len(z) else np.nan,
        "mean_diff": float(np.mean(x)-np.mean(z)) if (len(x) and len(z)) else np.nan,
        "Welch_t": float(t_stat),
        "Welch_p(one-sided)": float(t_p),
        "Welch_p(two-sided)": float(t_p_two),
        "MannWhitney_U": float(u_stat) if np.isfinite(u_stat) else np.nan,
        "MannWhitney_p(one-sided)": float(u_p) if np.isfinite(u_p) else np.nan,
        "Cohen_d": float(d),
        "Hedges_g": float(g),
        "Cliffs_delta": float(delta) if np.isfinite(delta) else np.nan,
        "ROC_AUC": float(roc_auc) if np.isfinite(roc_auc) else np.nan,
        "PR_AUC": float(pr_auc) if np.isfinite(pr_auc) else np.nan,
        **thr_pack,
        **boot_ci
    }

    # 图：箱线+散点；直方/核密
    try:
        import matplotlib.pyplot as plt
        os.makedirs(outdir, exist_ok=True)

        # 箱线+散点
        fig, ax = plt.subplots(figsize=(6,4))
        ax.boxplot([z, x], labels=["Historical","Recent"], showfliers=False)
        ax.set_title(f"{col}: group distribution")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"Figure_T3_box_{col}.png"), dpi=200)
        plt.close(fig)

        # 直方+核密
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(z, bins=40, alpha=0.5, density=True, label="Historical")
        ax.hist(x, bins=40, alpha=0.5, density=True, label="Recent")
        try:
            from scipy.stats import gaussian_kde
            grid = np.linspace(np.nanmin(s), np.nanmax(s), 256)
            if len(z)>3:
                kde0 = gaussian_kde(z); ax.plot(grid, kde0(grid), label="H-kde")
            if len(x)>3:
                kde1 = gaussian_kde(x); ax.plot(grid, kde1(grid), label="R-kde")
        except Exception:
            pass
        ax.legend(); ax.set_title(f"{col}: histogram/KDE"); ax.set_xlabel(col); ax.set_ylabel("density")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"Figure_T3_hist_{col}.png"), dpi=200)
        plt.close(fig)

    except Exception:
        pass

    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True, help="e.g. out_cum/scores_all.csv")
    ap.add_argument("--cols", nargs="+", default=["s_m","s_t","s_mtn","S"],
                    help="score columns to test")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--balanced", action="store_true",
                    help="downsample to 1:1 for the test (not modifying CSV)")
    ap.add_argument("--boot", type=int, default=0,
                    help="bootstrap rounds for CI of mean diff (0 to skip)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.scores_csv)

    rows = []
    for col in args.cols:
        if col not in df.columns:
            print(f"[Warn] skip missing column: {col}")
            continue
        row = run_one(df, col, args.outdir, pos_label="recent",
                      balanced=args.balanced, boot=args.boot, seed=args.seed)
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.outdir, "Table_T3_ttests.csv"), index=False)

    # 也提供 Markdown 版
    md_path = os.path.join(args.outdir, "Table_T3_ttests.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| score | N | mean_diff | Welch_p(one) | U_p(one) | d | g | CliffΔ | ROC_AUC | PR_AUC | Acc@thr |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['score_col']} | {r['N_total']} | {r['mean_diff']:.4f} | {r['Welch_p(one-sided)']:.3e} | "
                    f"{r['MannWhitney_p(one-sided)']:.3e} | {r['Cohen_d']:.3f} | {r['Hedges_g']:.3f} | "
                    f"{r['Cliffs_delta']:.3f} | {r['ROC_AUC']:.3f} | {r['PR_AUC']:.3f} | {r['Acc']:.3f} |\n")

    print("[OK] wrote:", os.path.join(args.outdir, "Table_T3_ttests.csv"),
          "and", os.path.join(args.outdir, "Table_T3_ttests.md"))
    print("[Note] Figures saved as Figure_T3_box_*.png / Figure_T3_hist_*.png")

if __name__ == "__main__":
    main()
