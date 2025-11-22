# eval_retrieval_pack.py
# 生成：retrieval_v3_perquery.csv, retrieval_v3_summary.json,
#      Figure_Retrieval_Bars.png, Figure_LiftK.png
import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt

def ndcg_at_k(labels, k=10):
    lab = np.asarray(labels[:k], float)
    gains = lab / np.log2(np.arange(2, len(lab)+2))
    dcg = gains.sum()
    # ideal
    ideal = np.sort(lab)[::-1]
    ideal_g = ideal / np.log2(np.arange(2, len(ideal)+2))
    idcg = ideal_g.sum() if ideal_g.sum() > 0 else 1.0
    return float(dcg / idcg)

def bootstrap_ci(vals, rounds=1000, seed=42, alpha=0.95):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0: return (np.nan, np.nan, np.nan)
    stats = [np.mean(v[rng.integers(0, n, n)]) for _ in range(rounds)]
    lo = (1 - alpha) / 2.0
    hi = 1 - lo
    return float(np.mean(v)), float(np.quantile(stats, lo)), float(np.quantile(stats, hi))

def main(outdir, scores_csv, topN=400, r_percent=20.0, include_self=True, seed=42):
    os.makedirs(outdir, exist_ok=True)
    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    b4_path   = os.path.join(outdir, "baseline_graph.csv")
    if not os.path.exists(pool_path): raise FileNotFoundError(pool_path)
    if not os.path.exists(scores_csv): raise FileNotFoundError(scores_csv)
    if not os.path.exists(b4_path): raise FileNotFoundError(b4_path)

    # 读 pool
    rows = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s: rows.append(json.loads(s))
    pool = pd.DataFrame(rows)
    if "field" not in pool.columns: pool["field"] = "ALL"
    if "text"  not in pool.columns: raise KeyError("pool_ALL.jsonl 需要包含 text 列（title+abstract）")
    S = pd.read_csv(scores_csv)[["id","S"]]
    B4 = pd.read_csv(b4_path)[["id","B4_graph_newedge"]]
    df = pool.merge(S, on="id", how="left").merge(B4, on="id", how="left")

    recent = df[df["split"]=="recent"].copy().reset_index(drop=True)
    if len(recent) == 0: raise RuntimeError("no recent docs")
    # 向量化 recent 文本，用于“nn_recent”候选池
    vec = TfidfVectorizer(max_features=60000)
    X = vec.fit_transform(recent["text"].astype(str).tolist())

    rng = np.random.default_rng(seed)
    P10, N10 = [], []
    perq = []
    r = r_percent

    for qi in range(len(recent)):
        qvec = X[qi]
        sims = (X @ qvec.T).toarray().ravel()  # 与所有 recent 的相似度
        order = np.argsort(-sims)  # 降序
        if not include_self:
            order = order[order != qi]
        order = order[:topN]
        cand = recent.iloc[order].copy()

        # pool 内按 B4_newedge 的 top-r% 作为正例（银标）
        thr = cand["B4_graph_newedge"].quantile(1.0 - r/100.0)
        labels = (cand["B4_graph_newedge"] >= thr).astype(int).to_numpy()
        if include_self:
            # self_or_top_r：把查询自身置为正例
            labels[np.where(order==qi)[0][0]] = 1

        # 用 \hat S 做排序的P@10 / nDCG@10（注意：这里评估的是“我们排序后Top10中真阳性的比例/质量”）
        cand_sorted = cand.sort_values("S", ascending=False).reset_index(drop=True)
        labels_sorted = labels[np.argsort(-cand["S"].to_numpy())]
        p10 = float(labels_sorted[:10].mean()) if len(labels_sorted) >= 10 else float(labels_sorted.mean())
        n10 = ndcg_at_k(labels_sorted, k=10)
        P10.append(p10); N10.append(n10)

        perq.append({
            "query_id": recent.loc[qi,"id"],
            "p_at_10": p10,
            "ndcg_at_10": n10
        })

    # 统计 & 置信区间
    p_mean, p_lo, p_hi = bootstrap_ci(P10, seed=seed)
    n_mean, n_lo, n_hi = bootstrap_ci(N10, seed=seed)
    summ = {
        "Q": int(len(recent)), "topN": int(topN), "r_percent": float(r),
        "include_self": bool(include_self),
        "P@10": float(p_mean), "P@10_CI": [float(p_lo), float(p_hi)],
        "nDCG@10": float(n_mean), "nDCG@10_CI": [float(n_lo), float(n_hi)],
        "baseline_random_P@10": float(r/100.0)
    }
    with open(os.path.join(outdir, "retrieval_v3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    pd.DataFrame(perq).to_csv(os.path.join(outdir,"retrieval_v3_perquery.csv"), index=False, encoding="utf-8")
    print("[OK] retrieval_v3 summary ->", summ)

    # ---- Figure: Bars with CI ----
    fig1 = os.path.join(outdir, "Figure_Retrieval_Bars.png")
    fig, ax = plt.subplots(figsize=(5.2,3.6), dpi=200)
    xs = np.array([0,1])
    means = [p_mean, n_mean]
    lows  = [p_mean - p_lo, n_mean - n_lo]
    highs = [p_hi - p_mean, n_hi - n_mean]
    ax.bar(xs, means, yerr=[lows, highs], capsize=4)
    ax.set_xticks(xs); ax.set_xticklabels(["P@10", "nDCG@10"])
    ax.set_ylim(0, 1.0)
    ax.axhline(r/100.0, ls="--")
    ax.set_title(f"Retrieval (topN={topN}, r%={r}, self={include_self})")
    ax.set_ylabel("score")
    fig.tight_layout(); plt.savefig(fig1); plt.close()
    print("[OK] wrote", fig1)

    # ---- Figure: Lift@k ----
    # 重新按 k=1..10 统计平均 P@k，并与随机基线比较
    lifts = []
    for k in range(1,11):
        vals = []
        for qi in range(len(recent)):
            qvec = X[qi]
            sims = (X @ qvec.T).toarray().ravel()
            order = np.argsort(-sims)
            if not include_self:
                order = order[order != qi]
            order = order[:topN]
            cand = recent.iloc[order].copy()
            thr = cand["B4_graph_newedge"].quantile(1.0 - r/100.0)
            labels = (cand["B4_graph_newedge"] >= thr).astype(int).to_numpy()
            if include_self:
                labels[np.where(order==qi)[0][0]] = 1
            labels_sorted = labels[np.argsort(-cand["S"].to_numpy())]
            pk = labels_sorted[:k].mean()
            vals.append(pk)
        lifts.append(np.mean(vals) / (r/100.0))

    fig2 = os.path.join(outdir, "Figure_LiftK.png")
    fig, ax = plt.subplots(figsize=(5.2,3.6), dpi=200)
    ax.plot(range(1,11), lifts, marker="o")
    ax.axhline(1.0, ls="--")
    ax.set_xlabel("k"); ax.set_ylabel("lift@k (= P@k / r%)")
    ax.set_title("Lift@k (higher is better)")
    ax.set_xticks(range(1,11))
    fig.tight_layout(); plt.savefig(fig2); plt.close()
    print("[OK] wrote", fig2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--topN", type=int, default=400)
    ap.add_argument("--r_percent", type=float, default=20.0)
    ap.add_argument("--include_self", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.outdir, args.scores_csv, args.topN, args.r_percent, args.include_self, args.seed)
