# -*- coding: utf-8 -*-
import argparse, os, json, yaml, math, numpy as np, pandas as pd
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
    return dcg / (ideal if ideal > 0 else 1.0)

def bootstrap_ci(vals, rounds=1000, seed=123):
    rng = np.random.default_rng(seed)
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    boots = []
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean([vals[i] for i in idx])))
    return {"mean": float(np.mean(vals)),
            "low":  float(np.percentile(boots, 2.5)),
            "high": float(np.percentile(boots, 97.5))}

def build_rarity_on_history(hist_df):
    vec_m = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    vec_t = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xm_hist = vec_m.fit_transform(hist_df["methods"].apply(list_to_text))
    Xt_hist = vec_t.fit_transform(hist_df["tasks"].apply(list_to_text))
    def rarity_row(row):
        mtxt = list_to_text(row["methods"])
        ttxt = list_to_text(row["tasks"])
        cm = 1.0 - float(cosine_similarity(vec_m.transform([mtxt]), Xm_hist).max()) if Xm_hist.shape[0] else 0.0
        ct = 1.0 - float(cosine_similarity(vec_t.transform([ttxt]), Xt_hist).max()) if Xt_hist.shape[0] else 0.0
        return 0.5*(cm+ct)
    return rarity_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", default=None)  # 可指定 meta/plus/base
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    outdir = args.outdir
    hist = pd.read_json(os.path.join(outdir, "historical.norm.jsonl"), lines=True)
    rec  = pd.read_json(os.path.join(outdir, "recent.norm.jsonl"), lines=True)
    all_pool = pd.read_json(os.path.join(outdir, "pool_ALL.jsonl"), lines=True)

    # 分数（默认 scores_all.csv；你现在用的是翻转后的 meta）
    scores_csv = args.scores_csv or os.path.join(outdir, "scores_all.csv")
    sc = pd.read_csv(scores_csv)
    if "id" not in sc.columns or "S" not in sc.columns:
        raise ValueError(f"{scores_csv} must contain columns: id, S")

    # 合并分数
    all_pool = all_pool.merge(sc[["id","S"]], on="id", how="left").fillna({"S":0.0})
    rec      = rec.merge(sc[["id","S"]],      on="id", how="left").fillna({"S":0.0})

    # 历史稀有度（银标构造用）
    rarity_fn = build_rarity_on_history(hist)
    if "_rarity" not in all_pool.columns:
        all_pool["_rarity"] = all_pool.apply(rarity_fn, axis=1)
    if "_rarity" not in rec.columns:
        rec["_rarity"] = rec.apply(rarity_fn, axis=1)

    # 配置项
    r = float(cfg.get("distance_top_percent", 15)) / 100.0  # 默认放宽到 15%
    topk = int(cfg.get("retrieval", {}).get("k", 10) or 10)
    include_self = bool(cfg.get("retrieval", {}).get("include_self_label", True))  # 默认 True
    label_mode = str(cfg.get("retrieval", {}).get("label_mode", "self_or_top_r")).lower()
    pool_mode  = str(cfg.get("retrieval", {}).get("pool_mode", "recent")).lower()  # recent / field_recent / all

    # 候选池构造函数（按 query）
    def build_candidates(qrow):
        qfield = qrow.get("field", "ALL")
        if pool_mode == "recent":
            cand = rec.copy()
        elif pool_mode == "field_recent" and "field" in rec.columns:
            cand = rec[rec["field"] == qfield].copy()
            if len(cand) == 0:  # 兜底
                cand = rec.copy()
        else:
            cand = all_pool.copy()
        return cand

    # 打银标（每个 query 的候选池内）
    def label_candidates(cand, qid):
        cand = cand.sort_values("_rarity", ascending=False)
        cutoff = max(1, int(len(cand) * r))
        labels = pd.Series(0, index=cand.index, dtype=int)
        if label_mode in ("top_r", "self_or_top_r"):
            labels.iloc[:cutoff] = 1
        if include_self or label_mode in ("self_only",):
            if qid in set(cand["id"].values):
                labels.loc[cand.index[cand["id"] == qid]] = 1
        if label_mode == "self_only":
            labels[cand["id"] != qid] = 0
        cand = cand.assign(label=labels.values)
        return cand

    # 逐 query 评测
    p10_list, ndcg_list = [], []
    details = []
    for _, q in rec.iterrows():
        qid = q["id"]
        cand = build_candidates(q)
        cand = label_candidates(cand, qid)
        # 用你的 S 排序统计
        cand = cand.sort_values("S", ascending=False)
        labs = cand["label"].astype(int).tolist()
        p10  = precision_at_k(labs, topk)
        ndcg = ndcg_at_k(labs, topk)
        p10_list.append(p10); ndcg_list.append(ndcg)
        details.append({"qid": qid, "P@10": p10, "nDCG@10": ndcg})

    # 统计 + CI
    rounds = int(cfg.get("bootstrap",{}).get("rounds", 1000))
    seed   = int(cfg.get("bootstrap",{}).get("seed", 123))
    p_ci = bootstrap_ci(p10_list, rounds=rounds, seed=seed)
    n_ci = bootstrap_ci(ndcg_list, rounds=rounds, seed=seed)

    out = {
        "metric": {"P@10": float(np.mean(p10_list)), "nDCG@10": float(np.mean(ndcg_list))},
        "ci95": {"P@10": p_ci, "nDCG@10": n_ci},
        "queries": int(len(p10_list)),
        "silver_label": f"distance-per-query({label_mode})",
        "distance_top_percent": int(r*100),
        "include_self_label": bool(include_self),
        "pool_mode": pool_mode
    }
    with open(os.path.join(outdir, "retrieval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.DataFrame(details).to_csv(os.path.join(outdir, "retrieval_per_query.csv"),
                                 index=False, encoding="utf-8")
    # 兼容旧制表
    pd.DataFrame([{"nDCG@10": out['metric']['nDCG@10'], "P@10": out['metric']['P@10']}]) \
      .to_csv(os.path.join(outdir, "retrieval.csv"), index=False)

    print(f"[OK] retrieval v2: P@10={out['metric']['P@10']:.4f}, nDCG@10={out['metric']['nDCG@10']:.4f}, "
          f"Q={out['queries']}, pool={pool_mode}, include_self={include_self}, label_mode={label_mode}")

if __name__ == "__main__":
    main()
