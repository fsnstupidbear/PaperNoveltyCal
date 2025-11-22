# tune_weights.py — robust version (no field required), learns weights -> scores_weighted.csv
import os, json, argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def ecdf_rank_norm(x_train, x_all):
    xs = np.sort(np.asarray(x_train, float))
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return np.zeros_like(x_all, float)
    pos = np.searchsorted(xs, np.asarray(x_all, float), side="right")
    return (pos - 0.5) / max(1, len(xs))

def winsorize(x, lo=0.01, hi=0.99):
    x = np.asarray(x, float)
    a, b = np.nanquantile(x, lo), np.nanquantile(x, hi)
    return np.clip(x, a, b)

def safe_read_pool_meta(outdir):
    """从 pool_ALL.jsonl 取 [id, split, field]，若没有 field 列则补 'ALL'。"""
    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    rows = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    meta = pd.DataFrame(rows)
    # 至少保证有 id / split
    need = []
    for k in ["id", "split"]:
        if k not in meta.columns:
            need.append(k)
    if need:
        raise KeyError(f"pool_ALL.jsonl is missing required columns: {need}")
    if "field" not in meta.columns:
        meta["field"] = "ALL"
    return meta[["id", "split", "field"]]

def main(outdir, r_percent=20.0, seed=42):
    os.makedirs(outdir, exist_ok=True)
    scores = pd.read_csv(os.path.join(outdir, "scores_all.csv"))
    if not set(["id","split","s_mn","s_tn","s_mtn"]).issubset(scores.columns):
        raise KeyError("scores_all.csv must contain columns: id, split, s_mn, s_tn, s_mtn")

    b4_path = os.path.join(outdir, "baseline_graph.csv")
    if not os.path.exists(b4_path):
        raise FileNotFoundError(f"{b4_path} not found. Run baselines_graph.py first.")
    b4 = pd.read_csv(b4_path)[["id","B4_graph_newedge"]]

    # 读 pool 元信息（补齐 field）
    meta = safe_read_pool_meta(outdir)

    # 合并
    df = scores.merge(b4, on="id", how="left").merge(meta, on=["id","split"], how="left")
    if "field" not in df.columns:
        # 双保险（理论上不会走到这里）
        df["field"] = "ALL"

    recent = df[df["split"]=="recent"].copy().reset_index(drop=True)
    hist   = df[df["split"]=="historical"].copy().reset_index(drop=True)
    if len(recent) == 0 or len(hist) == 0:
        raise RuntimeError("Need both recent and historical in scores_all.csv")

    # ====== 分位归一 + 去极值（按 field；若 field==ALL 就是单组） ======
    feats = ["s_mn","s_tn","s_mtn"]
    for fld in recent["field"].fillna("ALL").unique():
        idx_r = (recent["field"].fillna("ALL") == fld)
        idx_h = (hist["field"].fillna("ALL") == fld)

        for c in feats:
            rr = winsorize(recent.loc[idx_r, c].fillna(0.0))
            hr = winsorize(hist.loc[idx_h, c].fillna(0.0))
            # 用 recent 的经验分布做 ECDF 映射
            recent.loc[idx_r, c+"_p"] = ecdf_rank_norm(rr, rr)
            hist.loc[idx_h, c+"_p"]   = ecdf_rank_norm(rr, hr)

    # ====== 构造银标（global top-r% by newedge on recent） ======
    if recent["B4_graph_newedge"].notna().sum() == 0:
        raise RuntimeError("No B4_graph_newedge values for recent. Re-run baselines_graph.py")
    thr = recent["B4_graph_newedge"].quantile(1.0 - r_percent/100.0)
    y = (recent["B4_graph_newedge"] >= thr).astype(int).values
    X = recent[[f+"_p" for f in feats]].fillna(0.0).values

    # ====== 5 折 CV 学权重（LogReg 概率作为 \hat S） ======
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recent))
    coefs, intercepts, aucs = [], [], []
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y[va], oof[va]))
        coefs.append(clf.coef_[0].tolist()); intercepts.append(float(clf.intercept_[0]))

    # 用全部 recent 拟合一版，给 hist 产分
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=seed)
    clf.fit(X, y)
    hist_X = hist[[f+"_p" for f in feats]].fillna(0.0).values
    hist_pred = clf.predict_proba(hist_X)[:,1]

    # ====== 写出 scores_weighted.csv（两列 id,S + split 便于核查） ======
    out = pd.concat([
        pd.DataFrame({"id": recent["id"], "S": oof, "split": "recent"}),
        pd.DataFrame({"id": hist["id"],   "S": hist_pred, "split": "historical"})
    ], ignore_index=True)
    out = out[["id","S","split"]]
    out.to_csv(os.path.join(outdir,"scores_weighted.csv"), index=False, encoding="utf-8")

    report = {
        "r_percent": float(r_percent),
        "threshold": float(thr),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "coef_mean": list(np.mean(np.array(coefs), axis=0)),
        "intercept_mean": float(np.mean(intercepts)),
        "features": [f+"_p" for f in feats]
    }
    with open(os.path.join(outdir,"weights_report.json"),"w",encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[OK] wrote scores_weighted.csv & weights_report.json", report)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--r_percent", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.outdir, args.r_percent, args.seed)
