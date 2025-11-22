# eval_binary_global.py  —— 处理 NaN/非数字/无效 split 的稳健版
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score
)
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def bootstrap_ci(vals, rounds=1000, alpha=0.05):
    vals = np.asarray(vals); n = len(vals)
    bs = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        bs.append(np.mean(vals[idx]))
    lo = np.percentile(bs, 100*alpha/2)
    hi = np.percentile(bs, 100*(1-alpha/2))
    return float(lo), float(hi)

def main(outdir, scores_csv, score_col="s_mtn", alpha=0.05):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(scores_csv)

    if "split" not in df.columns:
        raise KeyError(f"'split' 不在 {scores_csv} 列中")
    if score_col not in df.columns:
        raise KeyError(f"'{score_col}' 不在 {scores_csv} 列中，试试 --score_col S")

    # 将字符串数字/异常值转为数值，非法的变成 NaN
    s = pd.to_numeric(df[score_col], errors="coerce")
    split = df["split"].astype(str)

    # 仅保留标注为 recent/historical 的样本
    mask_split = split.isin(["recent", "historical"])

    # 过滤 NaN/±inf
    s_np = s.to_numpy(dtype=float)
    mask_finite = np.isfinite(s_np)

    # 合并过滤条件
    mask = mask_split & mask_finite

    # 打印过滤统计
    total = len(df)
    kept = int(mask.sum())
    dropped = total - kept
    recent_kept = int(((split=="recent") & mask).sum())
    hist_kept   = int(((split=="historical") & mask).sum())
    print({
        "total": total, "kept": kept, "dropped": dropped,
        "kept_recent": recent_kept, "kept_historical": hist_kept,
        "score_col": score_col
    })

    df = df.loc[mask].copy()
    y = (df["split"]=="recent").astype(int).to_numpy()
    s = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)

    # 至少需要两个类别
    if y.sum()==0 or y.sum()==len(y):
        raise ValueError("过滤后只剩下一个类别，无法计算 AUC；请检查数据或换 score_col。")

    # 自动方向：若 AUC<0.5 则翻转
    auc_raw = roc_auc_score(y, s)
    flipped = False
    if auc_raw < 0.5:
        s = -s
        flipped = True
    auc = roc_auc_score(y, s)
    ap  = average_precision_score(y, s)

    # 选 Youden J 的阈值做点估计
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    t_idx = int(np.argmax(j))
    t_best = thr[t_idx]
    yhat = (s >= t_best).astype(int)

    acc = (yhat==y).mean()
    acc_ci = bootstrap_ci((yhat==y).astype(float))
    f1  = f1_score(y, yhat)
    prec= precision_score(y, yhat)
    rec = recall_score(y, yhat)

    print({
        "N": int(len(y)),
        "flipped": flipped,
        "ROC_AUC": float(auc),
        "PR_AUC": float(ap),
        "best_thr": float(t_best),
        "Acc": float(acc), "Acc_CI": acc_ci,
        "F1": float(f1), "Prec": float(prec), "Rec": float(rec)
    })

    # 画 ROC
    plt.figure(figsize=(5,4), dpi=200)
    plt.plot(fpr, tpr, label=f"{score_col} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--',label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC: new vs old (score={score_col})")
    plt.legend(); plt.tight_layout()
    fn = os.path.join(outdir, f"Figure_ROC_global_{score_col}.png")
    plt.savefig(fn); plt.close(); print("[OK] wrote", fn)

    # 画 PR
    pr_p, pr_r, _ = precision_recall_curve(y, s)
    plt.figure(figsize=(5,4), dpi=200)
    plt.plot(pr_r, pr_p)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR: new vs old (AP={ap:.3f})")
    plt.tight_layout()
    fn = os.path.join(outdir, f"Figure_PR_global_{score_col}.png")
    plt.savefig(fn); plt.close(); print("[OK] wrote", fn)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", default="out_cum/scores_all.csv")
    ap.add_argument("--score_col", default="s_mtn")
    args = ap.parse_args()
    main(args.outdir, args.scores_csv, args.score_col)
