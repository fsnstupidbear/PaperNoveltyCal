# train_pairwise_logit.py — learn pairwise logistic weights from calibrated scores (robust to NaN)
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DEF_FEATURES_ORDER = [
    # 主推：三大归一化分量 + 标定派生
    "s_mn","s_tn","s_mtn",
    "z_s_mn","z_s_tn","z_s_mtn",
    "lp_s_mn","lp_s_tn","lp_s_mtn",
    # 备选：原始三项 + 标定派生（存在才用）
    "s_m","s_t","s_mt",
    "z_s_m","z_s_t","z_s_mt",
    "lp_s_m","lp_s_t","lp_s_mt",
]

CORE_FEATURES = ["s_mn","s_tn","s_mtn"]  # 可选：只用三大分量

def pick_features(df, core_only=False):
    base = CORE_FEATURES if core_only else DEF_FEATURES_ORDER
    feats = [c for c in base if c in df.columns]
    if not feats:
        raise ValueError("No usable features found. Expect any of: " + ", ".join(base))
    return feats

def ensure_text(df):
    # 确保有 text 列；若无则由 title/abstract 兜底
    if "text" not in df.columns:
        title = df["title"].astype(str) if "title" in df.columns else ""
        abstract = df["abstract"].astype(str) if "abstract" in df.columns else ""
        df["text"] = (title + " " + abstract).str.strip()
    df["text"] = df["text"].fillna("")
    return df

def impute_block(df, feats, strategy="median", ref_df=None):
    """对 df[feats] 做缺失值填补。
       strategy='median'：用 ref_df(默认 hist) 的列中位数；若仍全 NaN -> 0
       strategy='zero'：直接填 0
    """
    out = df.copy()
    for f in feats:
        out[f] = pd.to_numeric(out[f], errors="coerce")
    if strategy == "median":
        if ref_df is None:
            ref_df = out
        med = {}
        for f in feats:
            m = pd.to_numeric(ref_df[f], errors="coerce").median()
            med[f] = 0.0 if pd.isna(m) else float(m)
        for f in feats:
            out[f] = out[f].fillna(med[f])
    else:
        out[feats] = out[feats].fillna(0.0)
    # 兜底：仍有 NaN 的置 0
    out[feats] = out[feats].fillna(0.0)
    return out

def build_pairs(df, feats, topk=8, match_by_field=True, max_features=60000, seed=42, impute_strategy="median"):
    np.random.seed(seed)

    df = ensure_text(df.copy())
    if "field" not in df.columns:
        df["field"] = "ALL"
    df["field"] = df["field"].fillna("ALL").replace("", "ALL")

    recent = df[df["split"]=="recent"].copy()
    hist   = df[df["split"]=="historical"].copy()
    if recent.empty or hist.empty:
        raise ValueError("recent 或 historical 为空，无法构造 pairwise 样本")

    # 先用历史分布做特征填补（更稳）
    hist = impute_block(hist, feats, strategy=impute_strategy, ref_df=hist)
    recent = impute_block(recent, feats, strategy=impute_strategy, ref_df=hist)

    X_pairs, y_pairs = [], []
    pair_count = 0

    use_field = match_by_field and (df["field"].nunique() > 1)
    fields = recent["field"].unique() if use_field else ["ALL"]

    for fld in fields:
        r_sub = recent if not use_field else recent[recent["field"]==fld]
        h_sub = hist   if not use_field else hist[hist["field"]==fld]
        if h_sub.empty:
            h_sub = hist

        # TF-IDF 近邻（按文本确定候选历史对）
        vec = TfidfVectorizer(max_features=max_features)
        Xh = vec.fit_transform(h_sub["text"].astype(str))
        Xr = vec.transform(r_sub["text"].astype(str))
        sim = cosine_similarity(Xr, Xh)  # [nr, nh]

        k = min(max(1, topk), Xh.shape[0])
        # 取每个 recent 的 top-k 历史近邻
        part_k = min(k-1, max(0, Xh.shape[0]-1))
        top_idx = np.argpartition(-sim, kth=part_k, axis=1)[:, :k]

        r_feat = r_sub[feats].astype(float).to_numpy()
        h_feat = h_sub[feats].astype(float).to_numpy()

        # 保障不会再有 NaN
        r_feat = np.nan_to_num(r_feat, nan=0.0)
        h_feat = np.nan_to_num(h_feat, nan=0.0)

        for i, idxs in enumerate(top_idx):
            r_vec = r_feat[i]
            for j in idxs:
                h_vec = h_feat[j]
                delta = r_vec - h_vec          # 正样本：recent - historical
                X_pairs.append(delta); y_pairs.append(1)
                X_pairs.append(-delta); y_pairs.append(0)  # 对称负样本
                pair_count += 1

    X = np.vstack(X_pairs).astype(np.float32)
    y = np.array(y_pairs, dtype=np.int32)

    # 最后再兜底一次
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)

    return X, y, pair_count, recent, hist

def fit_logit_cv(X, y, C_grid, folds=5, seed=42):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best_C, best_auc, aucs = None, -1.0, {}
    for C in C_grid:
        fold_aucs = []
        for tr, va in cv.split(Xs, y):
            clf = LogisticRegression(
                C=float(C), solver="liblinear", penalty="l2",
                max_iter=4000, class_weight=None
            )
            clf.fit(Xs[tr], y[tr])
            p = clf.predict_proba(Xs[va])[:,1]
            fold_aucs.append(roc_auc_score(y[va], p))
        aucs[str(C)] = float(np.mean(fold_aucs))
        if aucs[str(C)] > best_auc:
            best_auc, best_C = aucs[str(C)], C

    clf = LogisticRegression(
        C=float(best_C), solver="liblinear", penalty="l2",
        max_iter=4000, class_weight=None
    )
    clf.fit(Xs, y)

    return {
        "scaler": scaler,
        "clf": clf,
        "best_C": float(best_C),
        "cv_auc_mean": float(best_auc),
        "cv_auc_grid": aucs,
    }

def score_docs(df, feats, model_pack):
    scaler = model_pack["scaler"]
    clf    = model_pack["clf"]
    X_doc = df[feats].astype(float).fillna(0.0).to_numpy()
    Xs    = scaler.transform(X_doc)
    s_linear = (Xs @ clf.coef_.ravel()) + float(clf.intercept_[0])  # 线性分数
    p = clf.predict_proba(Xs)[:,1]
    return s_linear, p

def main(outdir, scores_csv, neg_k, C_grid, folds, seed, no_field_match, max_features, impute_strategy, core_only):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(scores_csv)
    if "id" not in df.columns or "split" not in df.columns:
        raise KeyError(f"{scores_csv} 必须包含列 id 与 split")

    feats = pick_features(df, core_only=core_only)
    print(f"[Info] using features: {feats}")

    X, y, pairs, recent, hist = build_pairs(
        df, feats, topk=neg_k, match_by_field=(not no_field_match),
        max_features=max_features, seed=seed, impute_strategy=impute_strategy
    )
    print(f"[Info] built {pairs} positive pairs -> training samples X={X.shape}, y_pos_rate={y.mean():.3f}")
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        print(f"[Warn] X still has {n_nan} NaNs; replacing with 0")
        X = np.nan_to_num(X, nan=0.0)

    pack = fit_logit_cv(X, y, C_grid=C_grid, folds=folds, seed=seed)
    print(f"[CV] best C={pack['best_C']}, mean AUC={pack['cv_auc_mean']:.4f}, grid={pack['cv_auc_grid']}")

    # 文档打分
    df_out = df.copy()
    s_linear, p = score_docs(df_out, feats, pack)
    df_out["S_pair"] = s_linear.astype(float)
    df_out["S"]      = df_out["S_pair"]  # 兼容旧评测脚本默认读取 S

    out_csv = os.path.join(outdir, "scores_pair.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # 导出权重
    coef = pack["clf"].coef_.ravel().tolist()
    inter = float(pack["clf"].intercept_[0])
    weights = {
        "features": feats,
        "coef": {f: float(w) for f, w in zip(feats, coef)},
        "intercept": inter,
        "best_C": pack["best_C"],
        "cv_auc_mean": pack["cv_auc_mean"],
        "cv_auc_grid": pack["cv_auc_grid"],
        "neg_k": int(neg_k),
        "folds": int(folds),
        "seed": int(seed),
        "match_by_field": not no_field_match,
        "max_features": int(max_features),
        "impute_strategy": impute_strategy,
        "training_pairs": int(pairs),
        "core_only": bool(core_only),
    }
    with open(os.path.join(outdir, "pairwise_weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {os.path.join(outdir,'pairwise_weights.json')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--neg_k", type=int, default=8, help="top-k nearest historical negatives per recent")
    ap.add_argument("--C_grid", nargs="+", type=float, default=[0.1,0.3,1,3,10,30])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_field_match", action="store_true", help="do NOT restrict negatives to same field")
    ap.add_argument("--max_features", type=int, default=60000)
    ap.add_argument("--impute_strategy", choices=["median","zero"], default="median")
    ap.add_argument("--core_only", action="store_true", help="use only ['s_mn','s_tn','s_mtn']")
    args = ap.parse_args()

    main(args.outdir, args.scores_csv, args.neg_k, args.C_grid, args.folds,
         args.seed, args.no_field_match, args.max_features, args.impute_strategy, args.core_only)
