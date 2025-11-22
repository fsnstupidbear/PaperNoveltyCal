# -*- coding: utf-8 -*-
import argparse, os, json, yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_norm(outdir):
    hist = pd.read_json(os.path.join(outdir, "historical.norm.jsonl"), lines=True)
    rec  = pd.read_json(os.path.join(outdir, "recent.norm.jsonl"), lines=True)
    all_df = pd.concat([hist.assign(split="historical"), rec.assign(split="recent")], ignore_index=True)
    return hist, rec, all_df

def parse_scores_json(s):
    try:
        if isinstance(s, dict): return s
        return json.loads(s) if isinstance(s, str) and s.strip() else {}
    except:
        return {}

def rows_to_weighted_dicts(series_list, scores_series_list):
    """series: 列是 list[str] 的实体; scores: 对应 JSON 字典"""
    dicts = []
    for ents, js in zip(*series_list, *scores_series_list):
        pass

def weighted_dicts(ents_series, scores_series, default_w=1.0):
    mats = []
    for ents, js in zip(ents_series, scores_series):
        d = {}
        if not isinstance(ents, list): ents = []
        js = parse_scores_json(js)
        for e in ents:
            w = js.get(e, default_w)
            # 累加（同实体多次出现）
            d[e] = d.get(e, 0.0) + float(w)
        mats.append(d)
    return mats

def max_sim_to_hist(X, X_hist):
    # 返回每个样本与历史库的最大余弦相似度
    sims = cosine_similarity(X, X_hist)
    return sims.max(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    outdir = args.outdir

    hist, rec, all_df = load_norm(outdir)

    base_scores = pd.read_csv(os.path.join(outdir, "scores_all.csv"))  # 已有 S、s_mn/s_tn/s_mtn
    base_scores = base_scores[["id","S","s_mn","s_tn","s_mtn"]]

    # ---- 用“实体+分数”构建加权向量（在历史上拟合）----
    # methods
    hist_m_dicts = weighted_dicts(hist["methods"], hist["method_scores"])
    rec_m_dicts  = weighted_dicts(rec["methods"],  rec["method_scores"])
    dv_m = DictVectorizer(sparse=True)
    Xm_hist = dv_m.fit_transform(hist_m_dicts)
    Xm_rec  = dv_m.transform(rec_m_dicts)

    # tasks
    hist_t_dicts = weighted_dicts(hist["tasks"], hist["task_scores"])
    rec_t_dicts  = weighted_dicts(rec["tasks"],  rec["task_scores"])
    dv_t = DictVectorizer(sparse=True)
    Xt_hist = dv_t.fit_transform(hist_t_dicts)
    Xt_rec  = dv_t.transform(rec_t_dicts)

    # 相似度→距离（1 - max_cos）
    s_m_hist = 1.0 - max_sim_to_hist(Xm_hist, Xm_hist) if Xm_hist.shape[0] else np.zeros(len(hist))
    s_t_hist = 1.0 - max_sim_to_hist(Xt_hist, Xt_hist) if Xt_hist.shape[0] else np.zeros(len(hist))
    s_m_rec  = 1.0 - max_sim_to_hist(Xm_rec,  Xm_hist) if Xm_hist.shape[0] else np.zeros(len(rec))
    s_t_rec  = 1.0 - max_sim_to_hist(Xt_rec,  Xt_hist) if Xt_hist.shape[0] else np.zeros(len(rec))

    # 归一化（在 hist+rec 合并范围上）
    s_m_all = np.concatenate([s_m_hist, s_m_rec])
    s_t_all = np.concatenate([s_t_hist, s_t_rec])
    scaler_m = MinMaxScaler()
    scaler_t = MinMaxScaler()
    s_m_all_n = scaler_m.fit_transform(s_m_all.reshape(-1,1)).ravel()
    s_t_all_n = scaler_t.fit_transform(s_t_all.reshape(-1,1)).ravel()

    s_mn_hist, s_mn_rec = s_m_all_n[:len(hist)], s_m_all_n[len(hist):]
    s_tn_hist, s_tn_rec = s_t_all_n[:len(hist)], s_t_all_n[len(hist):]

    # 组装到一个 DataFrame
    hist_ids = hist["id"].tolist()
    rec_ids  = rec["id"].tolist()
    df_plus = pd.DataFrame(
        {"id": hist_ids + rec_ids,
         "s_mn_plus": np.concatenate([s_mn_hist, s_mn_rec]),
         "s_tn_plus": np.concatenate([s_tn_hist, s_tn_rec])}
    )

    # 融合：S_plus = w_m*s_mn_plus + w_t*s_tn_plus + w_mt*s_mtn(沿用原来的组合项)
    w_m  = float(cfg.get("weights",{}).get("method_entity", 0.8))
    w_t  = float(cfg.get("weights",{}).get("task_entity",   0.8))
    w_mt = float(cfg.get("weights",{}).get("combo_pair",    2.5))

    merged = base_scores.merge(df_plus, on="id", how="left")
    merged["S_plus"] = w_m*merged["s_mn_plus"].fillna(0) + \
                       w_t*merged["s_tn_plus"].fillna(0) + \
                       w_mt*merged["s_mtn"].fillna(0)

    # 为了兼容评测脚本，导出文件里把列名 'S' 指向新的分数；原始 S 另存 S_base
    out = merged[["id","S_plus","S","s_mn","s_tn","s_mtn"]].copy()
    out = out.rename(columns={"S":"S_base"})
    out.insert(1, "S", out["S_plus"])  # 评测找 S

    out_path = os.path.join(outdir, "scores_all_plus.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("[OK] scoring_plus ->", out_path)
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
