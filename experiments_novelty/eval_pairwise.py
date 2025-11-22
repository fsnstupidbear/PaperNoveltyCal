# eval_pairwise_pack.py
# Pairwise ROC under "nearest-historical" matching per field (or ALL)
# Outputs:
#   - <outdir>/pairwise_summary.json   (acc/auc with 95% CI, #pairs)
#   - <outdir>/Figure4_ROC_pairwise.png

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def bootstrap_ci(vals, rounds=1000, seed=42, alpha=0.95):
    """Return (mean, lo, hi) for bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    stats = [np.mean(v[rng.integers(0, n, n)]) for _ in range(rounds)]
    lo_q = (1 - alpha) / 2.0
    hi_q = 1 - lo_q
    return float(np.mean(v)), float(np.quantile(stats, lo_q)), float(np.quantile(stats, hi_q))

def load_pool(pool_path):
    rows = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    df = pd.DataFrame(rows)
    # 必要列兜底
    if "field" not in df.columns:
        df["field"] = "ALL"
    if "text" not in df.columns:
        raise KeyError("pool_ALL.jsonl 中缺少 'text' 列（需为 title+abstract 拼接文本）")
    if "split" not in df.columns:
        raise KeyError("pool_ALL.jsonl 中缺少 'split' 列（需包含 'historical'/'recent'）")
    return df

def build_pairs_delta(df, max_features=60000, seed=42):
    """
    为每个 field：用 TF-IDF 在同 field 内为每篇 recent 找到最相似的一篇 historical，
    计算 delta = S_recent - S_historical(nn)
    返回拼接的所有 delta（一维 numpy 数组）
    """
    # 分组前先确保字段
    if "field" not in df.columns:
        df["field"] = "ALL"

    hist_all = df[df["split"] == "historical"].copy().reset_index(drop=True)
    recent_all = df[df["split"] == "recent"].copy().reset_index(drop=True)

    if len(hist_all) == 0 or len(recent_all) == 0:
        raise RuntimeError("Need both historical and recent documents.")

    deltas = []
    for fld, grp_r in recent_all.groupby(recent_all["field"].fillna("ALL")):
        grp_h = hist_all[hist_all["field"].fillna("ALL") == fld]
        if len(grp_r) == 0 or len(grp_h) == 0:
            # 若该 field 没有历史文献，跳过
            continue

        # 用历史文献拟合 TF-IDF 词表，再把 recent 映射进来（同一空间）
        vec = TfidfVectorizer(max_features=max_features)
        Xh = vec.fit_transform(grp_h["text"].astype(str).tolist())
        Xr = vec.transform(grp_r["text"].astype(str).tolist())

        # 相似度矩阵：nr x nh
        sims = (Xr @ Xh.T)  # 稀疏矩阵
        sims_dense = sims.toarray()  # 规模不大时直接转 dense，方便 argmax

        nn_idx = sims_dense.argmax(axis=1)          # 每篇 recent 的最近邻 historical 索引
        S_h = grp_h["S"].to_numpy()
        S_r = grp_r["S"].to_numpy()
        delta = S_r - S_h[nn_idx]
        deltas.append(delta)

    if len(deltas) == 0:
        raise RuntimeError("No pairs constructed. Check 'field'/'text' columns or data balance.")
    return np.concatenate(deltas, axis=0)

def main(outdir, scores_csv, alpha=0.95, seed=42, max_features=60000, rounds=1000):
    os.makedirs(outdir, exist_ok=True)
    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(pool_path)
    if not os.path.exists(scores_csv):
        raise FileNotFoundError(scores_csv)

    pool = load_pool(pool_path)
    # 只保留需要的列，避免 merge 踩坑
    keep_cols = [c for c in ["id", "title", "text", "year", "field", "split"] if c in pool.columns]
    pool = pool[keep_cols]

    S = pd.read_csv(scores_csv)[["id", "S"]]
    df = pool.merge(S, on="id", how="left")

    # 构造配对 delta
    delta = build_pairs_delta(df, max_features=max_features, seed=seed)

    # Accuracy（delta>0）及其 CI
    acc_vals = (delta > 0).astype(float)
    acc, acc_lo, acc_hi = bootstrap_ci(acc_vals, rounds=rounds, seed=seed, alpha=alpha)

    # ROC-AUC：对称扩增法
    scores = np.r_[delta, -delta]
    labels = np.r_[np.ones(len(delta)), np.zeros(len(delta))]
    auc = float(roc_auc_score(labels, scores))

    # AUC 的 bootstrap CI（对 delta 进行重采样）
    rng = np.random.default_rng(seed)
    auc_samples = []
    for _ in range(rounds):
        idx = rng.integers(0, len(delta), len(delta))
        s = np.r_[delta[idx], -delta[idx]]
        y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
        auc_samples.append(roc_auc_score(y, s))
    auc_lo = float(np.quantile(auc_samples, (1 - alpha) / 2.0))
    auc_hi = float(np.quantile(auc_samples, 1 - (1 - alpha) / 2.0))

    # 画 ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    figp = os.path.join(outdir, "Figure4_ROC_pairwise.png")
    plt.figure(figsize=(4.8, 4.0), dpi=200)
    plt.plot(fpr, tpr, label=f"Ours (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC on Pairwise Judging (nearest-historical protocol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figp)
    plt.close()

    # 保存 JSON 摘要
    outj = {
        "pairs": int(len(delta)),
        "acc": float(acc),
        "acc_ci": [float(acc_lo), float(acc_hi)],
        "auc": float(auc),
        "auc_ci": [float(auc_lo), float(auc_hi)],
        "protocol": "nearest-historical per field (TF-IDF)",
        "seed": int(seed),
        "rounds": int(rounds),
        "max_features": int(max_features),
        "scores_csv": os.path.basename(scores_csv)
    }
    with open(os.path.join(outdir, "pairwise_summary.json"), "w", encoding="utf-8") as f:
        json.dump(outj, f, ensure_ascii=False, indent=2)

    print("[OK] pairwise summary ->", outj)
    print("[OK] wrote ROC fig ->", figp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", required=True, help="e.g., out_cum/scores_weighted.csv")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=60000)
    ap.add_argument("--rounds", type=int, default=1000, help="bootstrap rounds")
    args = ap.parse_args()
    main(args.outdir, args.scores_csv, args.alpha, args.seed, args.max_features, args.rounds)
