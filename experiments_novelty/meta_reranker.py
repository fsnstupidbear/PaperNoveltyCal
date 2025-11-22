# -*- coding: utf-8 -*-
import argparse, os, json, yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_to_text(x):
    if isinstance(x, list):
        return " ; ".join([str(i) for i in x])
    return str(x) if isinstance(x, str) else ""

def rarity_proxy_builder(hist_df):
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
    ap.add_argument("--scores_csv", default=None, help="默认 outdir/scores_all_plus.csv 若存在，否则 scores_all.csv")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    outdir = args.outdir

    # 数据
    hist = pd.read_json(os.path.join(outdir, "historical.norm.jsonl"), lines=True)
    rec  = pd.read_json(os.path.join(outdir, "recent.norm.jsonl"), lines=True)

    # 分数特征（优先用 plus 版）
    default_scores = os.path.join(outdir, "scores_all_plus.csv")
    fallbacks = [os.path.join(outdir,"scores_all.csv")]
    scores_csv = args.scores_csv or (default_scores if os.path.exists(default_scores) else fallbacks[0])
    sc = pd.read_csv(scores_csv)

    # 只保留需要列
    cols_needed = ["id","S","s_mn","s_tn","s_mtn"]
    for c in cols_needed:
        if c not in sc.columns:
            raise ValueError(f"Column {c} missing in {scores_csv}")
    sc = sc[cols_needed].copy()

    # 构造稀有度银标
    rarity_fn = rarity_proxy_builder(hist)
    rec["_rarity"] = rec.apply(rarity_fn, axis=1)
    r = float(cfg.get("distance_top_percent", 10))/100.0
    cutoff = max(1, int(len(rec)*r))
    rec = rec.sort_values("_rarity", ascending=False)
    rec["label"] = 0
    rec.iloc[:cutoff, rec.columns.get_loc("label")] = 1
    rec = rec.sort_index()  # 恢复原顺序

    # 合并特征
    rec_feats = rec[["id"]].merge(sc, on="id", how="left")
    rec_feats["rarity"] = rec["_rarity"]

    # OOF 学习到排序（LogReg）
    X = rec_feats[["s_mn","s_tn","s_mtn","S","rarity"]].fillna(0.0).values
    y = rec["label"].astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(rec), dtype=float)
    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200)
        clf.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(X[val_idx])[:,1]
    # 在 full recent 上再拟合一个最终模型，用于给 historical/剩余打分
    final_clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200)
    final_clf.fit(X, y)

    # 给 hist/rec 产出分数
    hist_feats = hist[["id"]].merge(sc, on="id", how="left")
    # hist 的 rarity 用同一个构造器
    hist["_rarity"] = hist.apply(rarity_fn, axis=1)
    hist_feats["rarity"] = hist["_rarity"]
    X_hist = hist_feats[["s_mn","s_tn","s_mtn","S","rarity"]].fillna(0.0).values
    pred_hist = final_clf.predict_proba(X_hist)[:,1]
    pred_rec  = oof  # 使用 OOF，避免偏置

    out = pd.DataFrame({"id": list(hist["id"])+list(rec["id"]),
                        "S":  list(pred_hist)+list(pred_rec)})
    out_path = os.path.join(outdir, "scores_all_meta.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("[OK] meta_reranker ->", out_path)
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
