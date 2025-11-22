# scoring.py — 直接使用 data_prep 产出的 s_m/s_t/s_mt 做归一化与加权合成 S
import os, json, argparse
import pandas as pd
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

def load_cfg(path):
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Please `pip install pyyaml`.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def minmax_norm(x, lo, hi):
    x = x.astype(float)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        # 无法归一或零方差，退化到常数 0.5（避免爆 NaN）
        return pd.Series([0.5] * len(x), index=x.index, dtype=float)
    out = (x - lo) / (hi - lo)
    return out.clip(0.0, 1.0)

def main(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)

    # 读取 data_prep.py 产出的规范化 JSONL
    p_hist   = os.path.join(outdir, "historical.norm.jsonl")
    p_recent = os.path.join(outdir, "recent.norm.jsonl")
    if not (os.path.exists(p_hist) and os.path.exists(p_recent)):
        raise FileNotFoundError("historical.norm.jsonl / recent.norm.jsonl not found. Run data_prep.py first.")

    hist   = read_jsonl(p_hist)
    recent = read_jsonl(p_recent)

    # 必要列检查
    need_cols = ["id","title","abstract","text","year","field","split","s_m","s_t","s_mt"]
    for c in need_cols:
        if c not in hist.columns or c not in recent.columns:
            raise KeyError(f"Missing required column `{c}` in normalized jsonl. Check data_prep.py output.")

    # 统计缺失，便于排查
    miss_hist = hist[["s_m","s_t","s_mt"]].isna().mean().to_dict()
    miss_recent = recent[["s_m","s_t","s_mt"]].isna().mean().to_dict()
    print("[Check] missing ratio (hist):", miss_hist)
    print("[Check] missing ratio (recent):", miss_recent)

    # 用历史集做 min–max（一致的参考系）
    lo_m, hi_m   = hist["s_m"].min(skipna=True),   hist["s_m"].max(skipna=True)
    lo_t, hi_t   = hist["s_t"].min(skipna=True),   hist["s_t"].max(skipna=True)
    lo_mt, hi_mt = hist["s_mt"].min(skipna=True),  hist["s_mt"].max(skipna=True)

    def apply_norm(df):
        df = df.copy()
        df["s_mn"]  = minmax_norm(df["s_m"],  lo_m,  hi_m)
        df["s_tn"]  = minmax_norm(df["s_t"],  lo_t,  hi_t)
        df["s_mtn"] = minmax_norm(df["s_mt"], lo_mt, hi_mt)
        return df

    hist_n   = apply_norm(hist)
    recent_n = apply_norm(recent)

    # 读取权重
    W = cfg.get("weights", {}) if isinstance(cfg.get("weights", {}), dict) else {}
    w_m  = float(W.get("method", 1.0))
    w_t  = float(W.get("task",   1.0))
    w_mt = float(W.get("pair",   0.35))

    def apply_score(df):
        df = df.copy()
        df["S"] = w_m*df["s_mn"] + w_t*df["s_tn"] + w_mt*df["s_mtn"]
        return df

    hist_sc   = apply_score(hist_n)
    recent_sc = apply_score(recent_n)

    all_sc = pd.concat([hist_sc, recent_sc], ignore_index=True)

    # 重要：只保留评测所需字段（兼容后续 eval_* 脚本）
    keep_cols = [
        "id","title","year","field","split",
        "s_m","s_t","s_mt","s_mn","s_tn","s_mtn","S"
    ]
    all_sc = all_sc[keep_cols].reset_index(drop=True)

    # 保存
    out_csv = os.path.join(outdir, "scores_all.csv")
    all_sc.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote {out_csv}")

    # 另外给一份 meta（便于你做翻转或额外实验），结构保持一致
    out_meta = os.path.join(outdir, "scores_all_meta.csv")
    all_sc.rename(columns={"S": "S_base"}, inplace=True)
    all_sc["S"] = all_sc["S_base"]  # 冗余一列方便后续脚本读取
    all_sc.to_csv(out_meta, index=False, encoding="utf-8")
    print(f"[OK] wrote {out_meta}")

    # 打印前几行便于人工检查
    print(all_sc.head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    main(cfg, args.outdir)
