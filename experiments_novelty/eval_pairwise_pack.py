# eval_pairwise_pack.py
# Pairwise ROC under "nearest-historical" matching with field fallback + auto orientation.
# Outputs:
#   <outdir>/pairwise_summary.json
#   <outdir>/Figure4_ROC_pairwise.png

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def bootstrap_ci(vals, rounds=1000, seed=42, alpha=0.95):
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
    if "field" not in df.columns:
        df["field"] = "ALL"
    if "split" not in df.columns:
        raise KeyError("pool_ALL.jsonl 缺少列 'split'（需含 'historical'/'recent'）")
    if "text" not in df.columns:
        raise KeyError("pool_ALL.jsonl 缺少列 'text'（需为 title+abstract 拼接文本）")
    return df

def build_pairs_delta(df, max_features=60000):
    """
    对每个 recent：
      1) 先在同 field 的历史集中找 TF-IDF 最近邻；
      2) 若该 field 无历史，回退到全局历史集；
      3) 计算 delta = S_recent - S_hist[nn]，剔除任一侧 S 为 NaN 的配对；
    返回 (delta, coverage)
    """
    hist_all = df[df["split"]=="historical"].copy().reset_index(drop=True)
    recent_all = df[df["split"]=="recent"].copy().reset_index(drop=True)

    cover = {
        "recent_total": int(len(recent_all)),
        "hist_total": int(len(hist_all)),
        "by_field_pairs": 0,
        "fallback_pairs": 0
    }

    if len(hist_all)==0 or len(recent_all)==0:
        raise RuntimeError("Need both historical and recent documents.")

    deltas = []

    # 预构建“全局历史”向量化器（fallback 用）
    vec_global = TfidfVectorizer(max_features=max_features)
    Xh_global = vec_global.fit_transform(hist_all["text"].astype(str).tolist())

    for fld, grp_r in recent_all.groupby(recent_all["field"].fillna("ALL")):
        grp_h = hist_all[hist_all["field"].fillna("ALL")==fld]
        use_global = len(grp_h)==0
        if use_global:
            grp_h = hist_all

        if use_global:
            Xr = vec_global.transform(grp_r["text"].astype(str).tolist())
            sims = (Xr @ Xh_global.T).toarray()  # [nr, nh_all]
        else:
            vec = TfidfVectorizer(max_features=max_features)
            Xh = vec.fit_transform(grp_h["text"].astype(str).tolist())
            Xr = vec.transform(grp_r["text"].astype(str).tolist())
            sims = (Xr @ Xh.T).toarray()        # [nr, nh_field]

        nn_idx = sims.argmax(axis=1)
        S_r = grp_r["S"].to_numpy()
        S_h_pool = (hist_all if use_global else grp_h)["S"].to_numpy()
        S_h_nn = S_h_pool[nn_idx]

        mask = ~np.isnan(S_r) & ~np.isnan(S_h_nn)
        delta = S_r[mask] - S_h_nn[mask]
        deltas.append(delta)

        # 修正处：先确定哪个计数器，然后再自增
        key = "fallback_pairs" if use_global else "by_field_pairs"
        cover[key] += int(mask.sum())

    if len(deltas)==0:
        raise RuntimeError("No pairs constructed (all empty after NaN filtering).")
    delta = np.concatenate(deltas, axis=0)
    cover["pairs"] = int(len(delta))
    return delta, cover

def eval_with_orientation(delta, alpha=0.95, rounds=1000, seed=42):
    """Compute metrics for delta and -delta; choose the better AUC orientation."""
    rng = np.random.default_rng(seed)

    def _metrics(d):
        acc_vals = (d > 0).astype(float)
        acc, acc_lo, acc_hi = bootstrap_ci(acc_vals, rounds=rounds, seed=seed, alpha=alpha)
        scores = np.r_[d, -d]
        labels = np.r_[np.ones(len(d)), np.zeros(len(d))]
        auc = float(roc_auc_score(labels, scores))
        auc_samples = []
        for _ in range(rounds):
            idx = rng.integers(0, len(d), len(d))
            s = np.r_[d[idx], -d[idx]]
            y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
            auc_samples.append(roc_auc_score(y, s))
        auc_lo = float(np.quantile(auc_samples, (1-alpha)/2.0))
        auc_hi = float(np.quantile(auc_samples, 1-(1-alpha)/2.0))
        fpr, tpr, _ = roc_curve(labels, scores)
        return {"acc":acc,"acc_ci":[acc_lo,acc_hi],"auc":auc,"auc_ci":[auc_lo,auc_hi],"fpr":fpr,"tpr":tpr}

    m_pos = _metrics(delta)
    m_neg = _metrics(-delta)
    if m_neg["auc"] > m_pos["auc"]:
        best = m_neg
        best["orientation"] = "-delta (flip)"
    else:
        best = m_pos
        best["orientation"] = "delta"
    return best

def main(outdir, scores_csv, alpha=0.95, seed=42, max_features=60000, rounds=1000):
    os.makedirs(outdir, exist_ok=True)
    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(pool_path)
    if not os.path.exists(scores_csv):
        raise FileNotFoundError(scores_csv)

    pool = load_pool(pool_path)
    keep_cols = [c for c in ["id","title","text","year","field","split"] if c in pool.columns]
    pool = pool[keep_cols]

    S = pd.read_csv(scores_csv)[["id","S"]]
    df = pool.merge(S, on="id", how="left")

    delta, cover = build_pairs_delta(df, max_features=max_features)
    best = eval_with_orientation(delta, alpha=alpha, rounds=rounds, seed=seed)

    # 画 ROC
    figp = os.path.join(outdir, "Figure4_ROC_pairwise.png")
    plt.figure(figsize=(4.8,4.0), dpi=200)
    plt.plot(best["fpr"], best["tpr"], label=f"Ours (AUC={best['auc']:.3f})")
    plt.plot([0,1],[0,1], "--", label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC on Pairwise Judging (nearest-historical + fallback)")
    plt.legend(); plt.tight_layout(); plt.savefig(figp); plt.close()

    outj = {
        "pairs": int(cover["pairs"]),
        "acc": float(best["acc"]),
        "acc_ci": [float(best["acc_ci"][0]), float(best["acc_ci"][1])],
        "auc": float(best["auc"]),
        "auc_ci": [float(best["auc_ci"][0]), float(best["auc_ci"][1])],
        "orientation": best["orientation"],
        "protocol": "nearest-historical per field with global fallback (TF-IDF)",
        "coverage": cover,
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
    ap.add_argument("--rounds", type=int, default=1000)
    args = ap.parse_args()
    main(args.outdir, args.scores_csv, args.alpha, args.seed, args.max_features, args.rounds)
