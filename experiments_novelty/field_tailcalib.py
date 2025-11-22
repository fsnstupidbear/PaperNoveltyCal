# field_tailcalib.py — 按“历史窗&领域”做 Z-score 和右尾 p 标定（健壮版）
import argparse, os, math, bisect
import numpy as np
import pandas as pd

def ecdf_tail_p(sorted_arr, x):
    """右尾 p = 1 - F(x)，F 用 (<=x)/(n+1) 防止 0/1；sorted_arr 必须升序。"""
    if sorted_arr.size == 0 or pd.isna(x):
        return np.nan
    k = bisect.bisect_right(sorted_arr, x)
    F = k / (len(sorted_arr) + 1.0)
    return 1.0 - F

def coalesce(df, target, *candidates, default=""):
    """把若干候选列按顺序合并到 target；不存在的列自动忽略；空值用 default。"""
    if target not in df.columns:
        df[target] = np.nan
    for c in candidates:
        if c in df.columns:
            df[target] = df[target].fillna(df[c])
    df[target] = df[target].fillna(default)
    return df

def main(outdir, scores_csv):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(scores_csv)
    if "id" not in df.columns or "split" not in df.columns:
        raise KeyError(f"{scores_csv} 必须包含列 id 与 split")

    # 读入规范化文本，拿到 field / title / abstract -> text
    hist_path = os.path.join(outdir, "historical.norm.jsonl")
    rec_path  = os.path.join(outdir, "recent.norm.jsonl")
    if not (os.path.exists(hist_path) and os.path.exists(rec_path)):
        raise FileNotFoundError("缺少 historical.norm.jsonl / recent.norm.jsonl，请先运行 data_prep.py")

    hist_norm = pd.read_json(hist_path, lines=True)
    rec_norm  = pd.read_json(rec_path,  lines=True)
    text_df = pd.concat([hist_norm, rec_norm], ignore_index=True)

    # 需要的列兜底
    for c in ["id","field","title","abstract"]:
        if c not in text_df.columns:
            text_df[c] = ""
    text_df["field"]    = text_df["field"].fillna("").replace("", "ALL")
    text_df["title"]    = text_df["title"].fillna("")
    text_df["abstract"] = text_df["abstract"].fillna("")
    text_df["text_norm"] = (text_df["title"].astype(str) + " " + text_df["abstract"].astype(str)).str.strip()

    # 合并；保留原 df 列名不变，来自 norm 的列加 _norm 后缀
    df = df.merge(
        text_df[["id","field","title","text_norm"]],
        on="id", how="left", suffixes=("", "_norm")
    )

    # 统一三列：field / title / text（有则保留，缺则用 *_norm 兜底）
    df = coalesce(df, "field", "field", "field_norm", default="ALL")
    df["field"] = df["field"].replace("", "ALL")
    df = coalesce(df, "title", "title", "title_norm", default="")
    # scores_all 通常没有 text；用 text_norm 兜底
    df = coalesce(df, "text", "text", "text_norm", default="")

    # 清理可能混入的列
    for c in ["field_norm", "title_norm"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    feats = [c for c in ["s_m","s_t","s_mt","s_mn","s_tn","s_mtn"] if c in df.columns]
    if not feats:
        raise ValueError(f"{scores_csv} 中找不到任何 s_m/s_t/s_mt/s_mn/s_tn/s_mtn 列")

    hist = df[df["split"]=="historical"].copy()
    if hist.empty:
        raise ValueError("historical 样本为空，无法做历史窗标定")

    # 逐 field 统计历史分布
    stats = {}
    for fld, g in hist.groupby(hist["field"].fillna("ALL")):
        stats[fld] = {}
        for f in feats:
            vals = pd.to_numeric(g[f], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                mu, sd, arr = 0.0, 1.0, np.array([])
            else:
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 1.0
                if sd < 1e-8: sd = 1.0
                arr = np.sort(vals)
            stats[fld][f] = {"mu": mu, "sd": sd, "sorted": arr}

    # 应用标定（Z-score 与 -log10右尾p）
    for f in feats:
        z_col, p_col, lp_col = f"z_{f}", f"p_{f}", f"lp_{f}"
        z_list, p_list, lp_list = [], [], []
        for _, r in df.iterrows():
            fld = r["field"] if isinstance(r["field"], str) and r["field"] else "ALL"
            st  = stats.get(fld, {}).get(f, None) or stats.get("ALL", {}).get(f, None)
            val = pd.to_numeric(r[f], errors="coerce")
            if st is None:
                mu, sd, arr = 0.0, 1.0, np.array([])
            else:
                mu, sd, arr = st["mu"], st["sd"], st["sorted"]
            if pd.notna(val):
                z = (val - mu) / sd
                # 右尾 p：越大越新颖
                if arr.size:
                    # 经验CDF：<=x 的比例
                    k = np.searchsorted(arr, val, side="right")
                    F = k / (arr.size + 1.0)
                    p = 1.0 - F
                else:
                    p = np.nan
                lp = -math.log10(max(p, 1e-12)) if not pd.isna(p) else np.nan
            else:
                z, p, lp = np.nan, np.nan, np.nan
            z_list.append(z); p_list.append(p); lp_list.append(lp)
        df[z_col] = z_list; df[p_col] = p_list; df[lp_col] = lp_list

    out_csv = os.path.join(outdir, "scores_cali.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")
    prev_cols = ["id","split","field","title","text"] + [c for c in df.columns if c.startswith("z_") or c.startswith("lp_")]
    print(df[prev_cols].head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", default="out_cum/scores_all.csv")
    args = ap.parse_args()
    main(args.outdir, args.scores_csv)
