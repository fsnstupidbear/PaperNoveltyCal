# data_prep.py  —— 读取“两份CSV + F列字典”的最小改动版
import os, json, argparse, math, ast
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

def read_csv_any(path, has_header):
    if has_header:
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, header=None)

def parse_cum_dict(cell, key_method, key_task, key_pair):
    """
    F列是形如 "{'CumulativeMethodScore': 0.57, 'CumulativeTaskScore': 0.54, 'CumulativeMethodTaskPairScore': 0.65}"
    的字符串。这里用 ast.literal_eval 安全解析。
    """
    if pd.isna(cell):
        return np.nan, np.nan, np.nan
    s = str(cell).strip()
    if not s:
        return np.nan, np.nan, np.nan
    try:
        d = ast.literal_eval(s)
        sm = float(d.get(key_method, np.nan))
        st = float(d.get(key_task, np.nan))
        smt = float(d.get(key_pair, np.nan))
        return sm, st, smt
    except Exception:
        # 兜底：尽量把单引号替换成双引号再试一次
        try:
            s2 = s.replace("'", '"').replace("None", "null")
            d = json.loads(s2)
            sm = float(d.get(key_method, np.nan))
            st = float(d.get(key_task, np.nan))
            smt = float(d.get(key_pair, np.nan))
            return sm, st, smt
        except Exception:
            return np.nan, np.nan, np.nan

def build_norm_df(df, split, cfg_cols, field_default="ALL"):
    """
    从一份CSV构建规范化DataFrame，包含：
      id, title, abstract, text, year, field, split, s_m, s_t, s_mt
    """
    title_idx     = cfg_cols.get("title_idx", 0)
    abstract_idx  = cfg_cols.get("abstract_idx", -1)
    cum_json_idx  = cfg_cols.get("cum_json_idx", None)
    cum_keys      = cfg_cols.get("cum_keys", {})
    key_m = cum_keys.get("method", "CumulativeMethodScore")
    key_t = cum_keys.get("task", "CumulativeTaskScore")
    key_mt = cum_keys.get("pair", "CumulativeMethodTaskPairScore")

    # 取标题
    if title_idx >= 0:
        title = df.iloc[:, title_idx].astype(str).fillna("")
    else:
        title = pd.Series([""] * len(df))

    # 取摘要
    if abstract_idx >= 0 and abstract_idx < df.shape[1]:
        abstract = df.iloc[:, abstract_idx].astype(str).fillna("")
    else:
        abstract = pd.Series([""] * len(df))

    # 解析F列字典
    if cum_json_idx is None:
        raise ValueError("columns.cum_json_idx is required in config for cumulative-score CSVs.")
    sm_list, st_list, smt_list = [], [], []
    for v in df.iloc[:, cum_json_idx].values:
        sm, st, smt = parse_cum_dict(v, key_m, key_t, key_mt)
        sm_list.append(sm); st_list.append(st); smt_list.append(smt)

    out = pd.DataFrame({
        "title": title,
        "abstract": abstract,
        "text": (title.fillna("") + " " + abstract.fillna("")).str.strip(),
        "year": 0,                      # 没有年份，这里占位 0
        "field": field_default,
        "split": split,
        "s_m": pd.to_numeric(sm_list, errors="coerce"),
        "s_t": pd.to_numeric(st_list, errors="coerce"),
        "s_mt": pd.to_numeric(smt_list, errors="coerce"),
    })
    # 补一个简单的 id
    prefix = "R" if split == "recent" else "H"
    out.insert(0, "id", [f"{prefix}{i}" for i in range(len(out))])
    return out

def save_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)

    recent_csv = cfg["input"]["recent_csv"]
    hist_csv   = cfg["input"]["historical_csv"]
    cols_cfg   = cfg["columns"]
    has_header = bool(cols_cfg.get("has_header", False))
    field_default = cfg.get("misc", {}).get("field_default", "ALL")

    # 读两份CSV
    df_recent_raw = read_csv_any(recent_csv, has_header)
    df_hist_raw   = read_csv_any(hist_csv, has_header)

    # 规范化
    recent = build_norm_df(df_recent_raw, "recent", cols_cfg, field_default)
    hist   = build_norm_df(df_hist_raw,   "historical", cols_cfg, field_default)

    # 存 JSONL（供 scoring.py 等脚本读取）
    p_hist = os.path.join(outdir, "historical.norm.jsonl")
    p_recent = os.path.join(outdir, "recent.norm.jsonl")
    save_jsonl(hist, p_hist)
    save_jsonl(recent, p_recent)

    # 合并一个池（有些脚本会读取）
    pool = pd.concat([hist, recent], ignore_index=True)
    p_pool = os.path.join(outdir, "pool_ALL.jsonl")
    save_jsonl(pool, p_pool)

    # 也方便人工查看：存一份CSV
    pool_csv = os.path.join(outdir, "pool_ALL.preview.csv")
    pool.to_csv(pool_csv, index=False, encoding="utf-8")

    print(f"[OK] normalized -> {p_hist} , {p_recent} ; pool_ALL.jsonl")
    print(f"[OK] preview CSV -> {pool_csv}")
    # 顺手提示一下缺失率，帮你排查异常
    miss_ratio = pool[["s_m","s_t","s_mt"]].isna().mean().to_dict()
    print("[Check] missing ratio:", miss_ratio)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    main(cfg, args.outdir)
