# -*- coding: utf-8 -*-
import argparse, os, json, yaml, math, random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def list_to_text(x):
    if isinstance(x, list):
        return " ; ".join([str(i) for i in x])
    return str(x) if isinstance(x, str) else ""

def precision_at_k(labels, k=10):
    labs = np.array(labels[:k], dtype=int)
    return float(labs.sum()) / max(1, len(labs))

def dcg_at_k(labels, k=10):
    s = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel:
            s += 1.0 / math.log2(i + 1)
    return s

def ndcg_at_k(labels, k=10):
    dcg = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

def bootstrap_ci(vals, rounds=1000, seed=123, alpha=0.05):
    rnd = random.Random(seed)
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    boots = []
    for _ in range(rounds):
        samp = [vals[rnd.randrange(n)] for _ in range(n)]
        boots.append(float(np.mean(samp)))
    boots.sort()
    low = boots[int(alpha/2 * rounds)]
    high = boots[int((1-alpha/2) * rounds) - 1]
    return {"mean": float(np.mean(vals)), "low": float(low), "high": float(high)}

def build_rarity_proxy(hist_df):
    vec_m = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    vec_t = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xm_hist = vec_m.fit_transform(hist_df["methods"].apply(list_to_text))
    Xt_hist = vec_t.fit_transform(hist_df["tasks"].apply(list_to_text))
    def rarity(row):
        mtxt = list_to_text(row["methods"])
        ttxt = list_to_text(row["tasks"])
        cm = 1.0 - float(cosine_similarity(vec_m.transform([mtxt]), Xm_hist).max()) if Xm_hist.shape[0] else 0.0
        ct = 1.0 - float(cosine_similarity(vec_t.transform([ttxt]), Xt_hist).max()) if Xt_hist.shape[0] else 0.0
        return 0.5*(cm+ct)
    return rarity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", default=None, help="默认自动找 outdir/scores_all.csv")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    outdir = args.outdir
    hist = pd.read_json(os.path.join(outdir, "historical.norm.jsonl"), lines=True)
    rec  = pd.read_json(os.path.join(outdir, "recent.norm.jsonl"), lines=True)
    pool = pd.read_json(os.path.join(outdir, "pool_ALL.jsonl"), lines=True)

    scores_csv = args.scores_csv or os.path.join(outdir, "scores_all.csv")
    if not os.path.exists(scores_csv):
        raise FileNotFoundError(f"scores_all.csv not found: {scores_csv}")
    scores = pd.read_csv(scores_csv)
    if "id" not in scores.columns or "S" not in scores.columns:
        raise ValueError("scores_all.csv must contain columns: id, S")
    pool = pool.merge(scores[["id","S"]], on="id", how="left").fillna({"S":0.0})

    # ---- 银标准 ----
    if str(cfg.get("silver_label","distance")).lower() == "distance":
        rarity_fn = build_rarity_proxy(hist)
        pool["_rarity"] = pool.apply(rarity_fn, axis=1)
        r = float(cfg.get("distance_top_percent", 10)) / 100.0
        topn = max(1, int(len(pool)*r))
        pool = pool.sort_values("_rarity", ascending=False)
        pool["label"] = 0
        pool.iloc[:topn, pool.columns.get_loc("label")] = 1
        # 回到按 S 排序做评估
        pool = pool.sort_values("S", ascending=False)
    else:
        raise ValueError("Only silver_label=distance is supported by this script.")

    labels = pool["label"].astype(int).tolist()
    p10 = precision_at_k(labels, k=10)
    n10 = ndcg_at_k(labels, k=10)

    # bootstrap（对排名位置进行自助法重采样近似）
    rounds = int(cfg.get("bootstrap",{}).get("rounds", 1000))
    seed   = int(cfg.get("bootstrap",{}).get("seed", 123))
    # 用 block bootstrap 的粗略近似：随机打散后按相同长度采样
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    boot_p = []
    boot_n = []
    for _ in range(rounds):
        samp = rng.choice(idx, size=len(idx), replace=True)
        labs = [labels[i] for i in samp]
        boot_p.append(precision_at_k(labs, 10))
        boot_n.append(ndcg_at_k(labs, 10))
    p_ci = {"mean": float(np.mean(boot_p)), "low": float(np.percentile(boot_p, 2.5)), "high": float(np.percentile(boot_p, 97.5))}
    n_ci = {"mean": float(np.mean(boot_n)), "low": float(np.percentile(boot_n, 2.5)), "high": float(np.percentile(boot_n, 97.5))}

    out = {
        "metric": {"P@10": p10, "nDCG@10": n10},
        "ci95": {"P@10": p_ci, "nDCG@10": n_ci},
        "silver_label": "distance",
        "distance_top_percent": cfg.get("distance_top_percent", 10),
        "pool_size": len(pool)
    }
    with open(os.path.join(outdir, "retrieval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 兼容旧版表格脚本：再写一份 CSV
    pd.DataFrame([{"nDCG@10": n10, "P@10": p10}]).to_csv(os.path.join(outdir, "retrieval.csv"), index=False)

    print(f"[OK] retrieval done: P@10={p10:.4f}, nDCG@10={n10:.4f}; result -> {outdir}/retrieval_summary.json")

if __name__ == "__main__":
    main()
