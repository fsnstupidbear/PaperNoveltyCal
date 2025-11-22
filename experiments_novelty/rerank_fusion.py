# -*- coding: utf-8 -*-
import argparse, os, json, yaml, math, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def list_to_text(x):
    if isinstance(x, list): return " ; ".join(map(str, x))
    return str(x) if isinstance(x, str) else ""

def precision_at_k(labels, k=10):
    labs = np.array(labels[:k], dtype=int)
    return float(labs.sum()) / max(1, len(labs))

def dcg_at_k(labels, k=10):
    s = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel: s += 1.0 / math.log2(i + 1)
    return s

def ndcg_at_k(labels, k=10):
    dcg = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    return dcg / (ideal if ideal > 0 else 1.0)

def bootstrap_mean(vals, rounds=1000, seed=123):
    rng = np.random.default_rng(seed); vals = list(vals); n = len(vals)
    if n == 0: return {"mean":0.0,"low":0.0,"high":0.0}
    boots = []
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean([vals[i] for i in idx])))
    return {"mean": float(np.mean(vals)),
            "low":  float(np.percentile(boots, 2.5)),
            "high": float(np.percentile(boots, 97.5))}

def build_rarity_on_history(hist_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_m = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    vec_t = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xm_hist = vec_m.fit_transform(hist_df["methods"].apply(list_to_text))
    Xt_hist = vec_t.fit_transform(hist_df["tasks"].apply(list_to_text))
    def rarity_row(row):
        mtxt = list_to_text(row["methods"]); ttxt = list_to_text(row["tasks"])
        cm = 1.0 - float(cosine_similarity(vec_m.transform([mtxt]), Xm_hist).max()) if Xm_hist.shape[0] else 0.0
        ct = 1.0 - float(cosine_similarity(vec_t.transform([ttxt]), Xt_hist).max()) if Xt_hist.shape[0] else 0.0
        return 0.5*(cm+ct)
    return rarity_row

def zscore(x):
    x = np.asarray(x, dtype=float); mu = x.mean(); sd = x.std()
    return (x - mu) / (sd + 1e-9)

def run_once(cfg, outdir, scores_csv, lam, nn_top_n, r_percent, include_self, label_mode):
    # 读数据
    hist = pd.read_json(os.path.join(outdir, "historical.norm.jsonl"), lines=True)
    rec  = pd.read_json(os.path.join(outdir, "recent.norm.jsonl"),     lines=True)
    sc   = pd.read_csv(scores_csv)
    if "id" not in sc.columns or "S" not in sc.columns:
        raise ValueError(f"{scores_csv} must contain columns: id, S")
    rec = rec.merge(sc[["id","S"]], on="id", how="left").fillna({"S":0.0})

    # 稀有度
    if "_rarity" not in rec.columns:
        rec["_rarity"] = rec.apply(build_rarity_on_history(hist), axis=1)

    # TF-IDF 文本相似构候选池
    vec_q = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_text = vec_q.fit_transform((rec["title"].fillna("") + " " + rec["abstract"].fillna("")).tolist())

    topk = int(cfg.get("retrieval", {}).get("k", 10) or 10)
    p10_list, ndcg_list = [], []

    rec_min = rec[["id","S","_rarity","field"]].copy() if "field" in rec.columns else rec[["id","S","_rarity"]].copy()

    for qi, qrow in rec.iterrows():
        qid = qrow["id"]; qvec = X_text[qi]
        sims = cosine_similarity(qvec, X_text).ravel()
        nn_idx = np.argsort(-sims)[:max(nn_top_n, topk+1)]
        cand = rec_min.iloc[nn_idx].copy()

        # 银标（每池 top-r% by rarity；可含自身）
        cutoff = max(1, int(len(cand) * (r_percent/100.0)))
        cand = cand.sort_values("_rarity", ascending=False)
        lab = pd.Series(0, index=cand.index, dtype=int)
        if label_mode in ("top_r", "self_or_top_r"):
            lab.iloc[:cutoff] = 1
        if include_self or label_mode in ("self_only",):
            if qid in set(cand["id"].values):
                lab.loc[cand.index[cand["id"] == qid]] = 1
        if label_mode == "self_only":
            lab[cand["id"] != qid] = 0
        cand = cand.assign(label=lab.values)

        # 关键：无监督融合 S′ = z(S) + λ·z(rarity)（在“该候选池”内标准化）
        s_comb = zscore(cand["S"].values) + float(lam) * zscore(cand["_rarity"].values)
        cand = cand.assign(S_fused=s_comb).sort_values("S_fused", ascending=False)

        labs = cand["label"].astype(int).tolist()
        p10  = precision_at_k(labs, topk)
        ndcg = ndcg_at_k(labs, topk)
        p10_list.append(p10); ndcg_list.append(ndcg)

    return float(np.mean(p10_list)), float(np.mean(ndcg_list)), p10_list, ndcg_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", default=None)     # 可指定 base/plus/meta
    ap.add_argument("--lams", default="0,0.5,1,1.5,2")# 扫 λ
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    outdir = args.outdir
    scores_csv = args.scores_csv or os.path.join(outdir, "scores_all.csv")

    nn_top_n  = int(cfg.get("retrieval", {}).get("nn_top_n", 200))
    r_percent = int(cfg.get("distance_top_percent", 20))
    include_self = bool(cfg.get("retrieval", {}).get("include_self_label", True))
    label_mode   = str(cfg.get("retrieval", {}).get("label_mode", "self_or_top_r")).lower()

    tried = []
    for lam in [float(x) for x in args.lams.split(",")]:
        p10, ndcg, p_list, n_list = run_once(cfg, outdir, scores_csv, lam, nn_top_n, r_percent, include_self, label_mode)
        tried.append({"lam":lam,"P@10":p10,"nDCG@10":ndcg})

    df = pd.DataFrame(tried).sort_values(["nDCG@10","P@10"], ascending=False)
    best = df.iloc[0].to_dict()
    lam = best["lam"]

    # 再跑一次 best λ 拿 CI 并写出结果
    p10, ndcg, p_list, n_list = run_once(cfg, outdir, scores_csv, lam, nn_top_n, r_percent, include_self, label_mode)
    ci_p  = bootstrap_mean(p_list, rounds=int(cfg.get("bootstrap",{}).get("rounds",1000)),
                           seed=int(cfg.get("bootstrap",{}).get("seed",123)))
    ci_nd = bootstrap_mean(n_list, rounds=int(cfg.get("bootstrap",{}).get("rounds",1000)),
                           seed=int(cfg.get("bootstrap",{}).get("seed",123)))

    out = {
        "metric": {"P@10": p10, "nDCG@10": ndcg},
        "ci95": {"P@10": ci_p, "nDCG@10": ci_nd},
        "queries": len(p_list),
        "pool_mode": f"nn_recent(topN={nn_top_n})",
        "distance_top_percent": r_percent,
        "include_self_label": include_self,
        "label_mode": label_mode,
        "lambda": lam,
        "scores_csv": os.path.basename(scores_csv)
    }
    with open(os.path.join(outdir, "retrieval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 兼容主表
    pd.DataFrame([{"nDCG@10": ndcg, "P@10": p10}]).to_csv(os.path.join(outdir, "retrieval.csv"), index=False)

    print(f"[OK] fusion rerank: lam={lam}, P@10={p10:.4f}, nDCG@10={ndcg:.4f} | pool=nn_recent({nn_top_n}), r%={r_percent}, self={include_self}")
    print("[Grid tried]\n", df.to_string(index=False))

if __name__ == "__main__":
    main()
