# eval_retrieval_v3.py — retrieval eval without `methods/tasks`
# Pool: TF-IDF nearest neighbors on `text` within RECENT set
# Labels: self_or_top_r using B4_graph_newedge (from baseline_graph.csv)
# Ranker: your novelty score S (from scores_all.csv)

import os, json, argparse, math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import yaml
except Exception:
    yaml = None

def load_cfg(path):
    if yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            rows.append(json.loads(s))
    return pd.DataFrame(rows)

def tokenize(s: str):
    # 轻量英文 tokenizer；中文可直接按字符/词切分也能跑
    import re
    if not isinstance(s, str): return []
    return [w.lower() for w in re.findall(r"[A-Za-z]+", s)]

def precision_at_k(labels, k=10):
    if len(labels) == 0: return 0.0
    k = min(k, len(labels))
    return float(np.sum(labels[:k])) / float(k)

def dcg_at_k(labels, k=10):
    k = min(k, len(labels))
    labels = np.asarray(labels[:k], dtype=float)
    if k == 0: return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum(labels * discounts))

def ndcg_at_k(labels, k=10):
    dcg = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    return 0.0 if ideal <= 0 else float(dcg / ideal)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", default=None, help="optional path to scores csv (default: outdir/scores_all.csv)")
    ap.add_argument("--topN", type=int, default=300)
    ap.add_argument("--r_percent", type=float, default=20.0)
    ap.add_argument("--include_self", action="store_true", default=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"{pool_path} not found. Run data_prep.py first.")
    df = read_jsonl(pool_path)
    if "text" not in df.columns or "split" not in df.columns:
        raise KeyError("pool_ALL.jsonl must contain `text` and `split` columns.")

    # 只在 RECENT 集内做检索评测
    recent = df[df["split"]=="recent"].copy().reset_index(drop=True)
    Q = len(recent)
    if Q == 0:
        raise RuntimeError("No recent docs found.")

    # 读取你的分数 S
    scores_csv = args.scores_csv or os.path.join(outdir, "scores_all.csv")
    S = pd.read_csv(scores_csv)[["id","S"]]
    recent = recent.merge(S, on="id", how="left")
    if recent["S"].isna().all():
        raise RuntimeError("No S scores found for recent docs. Check scores_all.csv")

    # 读取 graph 基线用于打标（银标准）：new-edge 越大越“新”
    b4_path = os.path.join(outdir, "baseline_graph.csv")
    if not os.path.exists(b4_path):
        raise FileNotFoundError(f"{b4_path} not found. Run baselines_graph.py first.")
    B4 = pd.read_csv(b4_path)[["id","B4_graph_newedge"]]
    recent = recent.merge(B4, on="id", how="left")
    recent["B4_graph_newedge"] = recent["B4_graph_newedge"].fillna(0.0)

    # TF-IDF 邻居池（仅基于 RECENT 文本；避免历史信息泄漏）
    vec = TfidfVectorizer(analyzer=tokenize, max_features=40000, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    X = vec.fit_transform(recent["text"].astype(str).tolist())  # (Q, V)，L2 归一后 X * X.T 即余弦
    sim = (X @ X.T).toarray()  # Q x Q

    topN = max(1, int(args.topN))
    r_pct = max(0.0, min(100.0, float(args.r_percent)))
    Ks = 10  # 评测 @10

    P_list, N_list = [], []  # 每个 query 的P@10、nDCG@10
    valid_q = 0

    for i in range(Q):
        # 候选池：最近邻 topN
        order = np.argsort(-sim[i])  # 降序
        if not args.include_self:
            order = order[order != i]
        top_idx = order[:topN] if len(order) > topN else order

        # 候选的排名分数：采用你的 S
        cand_ids = recent.iloc[top_idx]["id"].tolist()
        cand_S   = recent.iloc[top_idx]["S"].values

        # 打标：self_or_top_r —— 自己为正；其余从 B4_graph_newedge 取 top r%
        cand_newedge = recent.iloc[top_idx]["B4_graph_newedge"].values
        k_pos = max(1, int(math.floor(len(top_idx) * (r_pct / 100.0))))
        # 排序找出阈值
        sorted_ne = np.sort(cand_newedge)[::-1]
        thr = sorted_ne[k_pos-1] if len(sorted_ne) >= k_pos else (sorted_ne[-1] if len(sorted_ne)>0 else 1.0)

        labels = (cand_newedge >= thr).astype(int)
        # include_self=True 则把自身置为正例
        if args.include_self:
            # 如果包含了自己，找到自己在 top_idx 中的位置
            self_pos = np.where(top_idx == i)[0]
            if len(self_pos) > 0:
                labels[self_pos[0]] = 1

        # 若这一池全为 0，跳过该 query（无银标准正例）
        if labels.sum() == 0:
            continue

        # 用 S 排序并计算指标
        rank_ord = np.argsort(-cand_S)  # S 越大越靠前
        ranked_labels = labels[rank_ord]

        P_list.append(precision_at_k(ranked_labels, k=Ks))
        N_list.append(ndcg_at_k(ranked_labels, k=Ks))
        valid_q += 1

    mean_P = float(np.mean(P_list)) if len(P_list)>0 else 0.0
    mean_N = float(np.mean(N_list)) if len(N_list)>0 else 0.0

    meta = {
        "P@10": mean_P,
        "nDCG@10": mean_N,
        "Q": valid_q,
        "pool": f"nn_recent(topN={topN})",
        "r%": r_pct,
        "include_self": bool(args.include_self),
        "label_mode": "self_or_top_r(B4_newedge)"
    }

    # 写 JSON 和 CSV（便于 make_table_main 读取）
    with open(os.path.join(outdir, "retrieval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    pd.DataFrame([{"P@10": mean_P, "nDCG@10": mean_N}]).to_csv(
        os.path.join(outdir, "retrieval_summary.csv"), index=False, encoding="utf-8"
    )

    print(f"[OK] retrieval v3: P@10={mean_P:.4f}, nDCG@10={mean_N:.4f}, Q={valid_q}, pool=nn_recent(topN={topN}), r%={r_pct}, include_self={args.include_self}, label_mode=self_or_top_r(B4_newedge)")

if __name__ == "__main__":
    main()
