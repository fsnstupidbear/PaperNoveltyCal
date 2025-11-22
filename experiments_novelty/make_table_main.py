# -*- coding: utf-8 -*-
import os, argparse, yaml, json, pandas as pd

def safe_read_csv(path, fallback=None):
    if os.path.exists(path):
        return pd.read_csv(path)
    return fallback

def main(cfg, outdir):
    # 读取基线
    b1 = safe_read_csv(os.path.join(outdir, "baselines_lexical.csv"))
    b3 = safe_read_csv(os.path.join(outdir, "baselines_topic.csv"))
    b4 = safe_read_csv(os.path.join(outdir, "baselines_graph.csv"))

    # 读取我们的方法的 pairwise 与检索
    pair = {}
    psum = os.path.join(outdir, "pairwise_summary.json")
    if os.path.exists(psum):
        pair = json.load(open(psum, "r", encoding="utf-8"))

    # 检索：base（干净）与 fusion（上界）
    retr_base = safe_read_csv(os.path.join(outdir, "retrieval_base.csv"), pd.DataFrame([{"nDCG@10": None, "P@10": None}]))
    retr_fus  = safe_read_csv(os.path.join(outdir, "retrieval.csv"),      pd.DataFrame([{"nDCG@10": None, "P@10": None}]))

    rows = []

    # 基线汇总（若存在）
    if b1 is not None:
        # 这里假定 baselines_lexical.csv 里已含聚合列（否则可自行改为读取 summary）
        rows.append(["TF-IDF/BM25", None, None, None, None])
    if b3 is not None:
        rows.append(["Topic-rarity (LDA)", None, None, None, None])
    if b4 is not None:
        rows.append(["Graph rarity (PMI)", None, None, None, None])

    # 我们的方法（分三行）
    rows.append(["Ours (PAIR, meta)", pair.get("accuracy"), pair.get("AUC"), None, None])
    rows.append(["Ours (RET, base)", None, None,
                 retr_base["nDCG@10"].iloc[0] if "nDCG@10" in retr_base.columns else None,
                 retr_base["P@10"].iloc[0]    if "P@10"    in retr_base.columns else None])
    rows.append(["Ours (RET, fusion-UB)", None, None,
                 retr_fus["nDCG@10"].iloc[0] if "nDCG@10" in retr_fus.columns else None,
                 retr_fus["P@10"].iloc[0]    if "P@10"    in retr_fus.columns else None])

    tab = pd.DataFrame(rows, columns=["Method","Accuracy","AUC","nDCG@10","Precision@10"])
    tab.to_csv(os.path.join(outdir, "Table_4_main.csv"), index=False, encoding="utf-8")
    # 也写 Markdown
    md = ["| Method | Accuracy | AUC | nDCG@10 | Precision@10 |",
          "|---|---:|---:|---:|---:|"]
    for _, r in tab.iterrows():
        md.append(f"| {r['Method']} | {r['Accuracy'] if pd.notnull(r['Accuracy']) else ''} | "
                  f"{r['AUC'] if pd.notnull(r['AUC']) else ''} | "
                  f"{r['nDCG@10'] if pd.notnull(r['nDCG@10']) else ''} | "
                  f"{r['Precision@10'] if pd.notnull(r['Precision@10']) else ''} |")
    open(os.path.join(outdir, "Table_4_main.md"), "w", encoding="utf-8").write("\n".join(md))
    print("[OK] Wrote out_real/Table_4_main.csv and out_real/Table_4_main.md")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    main(cfg, args.outdir)
