# eval_pairwise_nn.py  — nearest-historical pairing for pairwise judging
# 修复点：
# 1) 若 pool_ALL.jsonl 无 field，则自动补 "ALL"
# 2) 稀疏矩阵切片一律用 numpy 行索引（np.flatnonzero），不再用 pandas Series 直接布尔索引
# 3) 文本向量化用 title+abstract 的 text 字段；若缺失，报错提示

import os, json, argparse, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def bootstrap_ci(vals, rounds=1000, seed=42, alpha=0.95):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    stats = [np.mean(v[rng.integers(0, n, n)]) for _ in range(rounds)]
    lo = (1 - alpha) / 2.0
    hi = 1 - lo
    return (float(np.mean(v)),
            float(np.quantile(stats, lo)),
            float(np.quantile(stats, hi)))

def main(config, outdir, scores_csv, rounds=1000, seed=42, alpha=0.95):
    os.makedirs(outdir, exist_ok=True)

    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"{pool_path} not found. Run data_prep.py first.")

    pool = read_jsonl(pool_path)
    if "id" not in pool.columns or "split" not in pool.columns:
        raise KeyError("pool_ALL.jsonl must contain 'id' and 'split' columns.")
    if "text" not in pool.columns:
        raise KeyError("pool_ALL.jsonl must contain 'text' (title+abstract) column for TF-IDF.")
    if "field" not in pool.columns:
        pool["field"] = "ALL"  # 兜底一个领域列

    S = pd.read_csv(scores_csv)[["id", "S"]]

    # 合并分数
    df = pool.merge(S, on="id", how="left")

    # 按 split 拆分并重置索引，保持行顺序与 X 子矩阵一致
    hist   = df[df["split"] == "historical"].copy().reset_index(drop=True)
    recent = df[df["split"] == "recent"].copy().reset_index(drop=True)

    if len(hist) == 0 or len(recent) == 0:
        raise RuntimeError("Need both historical and recent docs for pairwise evaluation.")

    # 在整个 df 的顺序上建立 TF-IDF；随后用位置切片取子矩阵
    vec = TfidfVectorizer(max_features=60000)
    X_all = vec.fit_transform(df["text"].astype(str).tolist())

    # 记录 df 中 hist/recent 的“原始位置”索引，以便从 X_all 切分
    idx_df_hist = np.flatnonzero(df["split"].to_numpy() == "historical")
    idx_df_recent = np.flatnonzero(df["split"].to_numpy() == "recent")

    X_hist = X_all[idx_df_hist]
    X_recent = X_all[idx_df_recent]

    pairs = []

    # 逐领域配对；若只有 ALL，则正好一组
    recent_fields = recent["field"].fillna("ALL").to_numpy()
    hist_fields   = hist["field"].fillna("ALL").to_numpy()

    unique_fields = pd.unique(recent_fields)
    for fld in unique_fields:
        mask_r = (recent_fields == fld)
        mask_h = (hist_fields   == fld)

        if not np.any(mask_r) or not np.any(mask_h):
            continue

        idx_r = np.flatnonzero(mask_r)  # recent 子集在 recent 中的行号
        idx_h = np.flatnonzero(mask_h)  # hist 子集在 hist 中的行号

        Xr = X_recent[idx_r]
        Xh = X_hist[idx_h]

        # 余弦相似 = L2 归一 TF-IDF 的内积；直接稀疏乘积
        sim = (Xr @ Xh.T).toarray()  # 形状: [len(idx_r), len(idx_h)]

        # 为每个 recent 样本，找一个历史最近邻作为对手
        nn_idx = sim.argmax(axis=1)  # 在 idx_h 范围内的列索引
        for pos_in_r, pos_in_h in enumerate(nn_idx):
            r_row = recent.iloc[idx_r[pos_in_r]]
            h_row = hist.iloc[idx_h[pos_in_h]]
            # 记录 (S_recent, S_hist)
            pairs.append((float(r_row["S"]), float(h_row["S"])))

    recent_s = np.array([p[0] for p in pairs], dtype=float)
    hist_s   = np.array([p[1] for p in pairs], dtype=float)
    if len(recent_s) == 0:
        raise RuntimeError("No field-overlapping pairs constructed. Check your 'field' distribution.")

    diff = recent_s - hist_s
    acc, acc_lo, acc_hi = bootstrap_ci((diff > 0).astype(float), rounds, seed, alpha)

    # AUC 计算：把差值 diff 当作正类分数，镜像一份负类
    score = np.concatenate([diff, -diff])
    label = np.concatenate([np.ones(len(diff)), np.zeros(len(diff))])
    auc = float(roc_auc_score(label, score))

    rng = np.random.default_rng(seed)
    auc_bs = []
    for _ in range(rounds):
        idx = rng.integers(0, len(diff), len(diff))
        s = np.concatenate([diff[idx], -diff[idx]])
        y = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
        auc_bs.append(roc_auc_score(y, s))
    lo = (1 - alpha) / 2.0
    hi = 1 - lo

    out = {
        "acc": float(acc),
        "acc_ci": [float(acc_lo), float(acc_hi)],
        "auc": float(auc),
        "auc_ci": [float(np.quantile(auc_bs, lo)), float(np.quantile(auc_bs, hi))],
        "pairs": int(len(diff))
    }
    with open(os.path.join(outdir, "pairwise_nn_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[Pairwise-NN]", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.config, args.outdir, args.scores_csv, args.rounds, args.seed)
