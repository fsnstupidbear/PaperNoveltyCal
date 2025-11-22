# baselines_topic.py — LDA topic-rarity baseline using only `text` (title+abstract).
# Compatible with the cumulative-score pipeline (no `methods`/`tasks` columns needed).

import os, json, argparse, re
import pandas as pd

try:
    from gensim import corpora, models
except Exception as e:
    raise RuntimeError("gensim is required. Please `pip install gensim`.") from e

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
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return pd.DataFrame(rows)

_TOKEN = re.compile(r"[A-Za-z]+")

def tokenize(txt: str):
    if not isinstance(txt, str):
        return []
    return [w.lower() for w in _TOKEN.findall(txt)]

def main(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)

    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"{pool_path} not found. Run data_prep.py first.")

    df = read_jsonl(pool_path)
    if "text" not in df.columns or "split" not in df.columns:
        raise KeyError("pool_ALL.jsonl must contain `text` and `split` columns ( produced by data_prep.py ).")

    hist = df[df["split"] == "historical"].copy()
    recent = df[df["split"] == "recent"].copy()

    # LDA config with safe defaults (works even if `lda:` is missing in config)
    lda_cfg = cfg.get("lda", {}) if isinstance(cfg.get("lda", {}), dict) else {}
    num_topics   = int(lda_cfg.get("num_topics", 100))
    passes       = int(lda_cfg.get("passes", 5))
    iterations   = int(lda_cfg.get("iterations", 200))
    random_state = int(lda_cfg.get("random_state", 42))
    no_below     = int(lda_cfg.get("no_below", 5))
    no_above     = float(lda_cfg.get("no_above", 0.5))

    # Tokenize
    docs_hist  = [tokenize(t) for t in hist["text"].astype(str).tolist()]
    docs_recent = [tokenize(t) for t in recent["text"].astype(str).tolist()]

    # Dictionary from historical only (avoid leakage)
    dict_ = corpora.Dictionary(docs_hist)
    if len(dict_) == 0:
        # Degenerate: write a flat baseline and exit
        out_csv = os.path.join(outdir, "baseline_topic.csv")
        recent_out = recent[["id","title","year","field"]].copy()
        recent_out["B3_topic_pair_rarity"] = 0.5
        recent_out.to_csv(out_csv, index=False, encoding="utf-8")
        print("[WARN] Empty dictionary; wrote constant rarity=0.5")
        print(f"[OK] wrote {out_csv}")
        return

    dict_.filter_extremes(no_below=no_below, no_above=no_above)
    corpus_hist = [dict_.doc2bow(d) for d in docs_hist]

    # Train LDA on historical corpus
    lda = models.LdaModel(
        corpus=corpus_hist,
        id2word=dict_,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        random_state=random_state
    )

    def dominant_topic(tokens):
        bow = dict_.doc2bow(tokens)
        if not bow:
            return -1
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        return max(dist, key=lambda x: x[1])[0] if dist else -1

    # Dominant topic for each doc
    hist["dom_topic"] = [dominant_topic(toks) for toks in docs_hist]
    recent["dom_topic"] = [dominant_topic(toks) for toks in docs_recent]

    # Rarity: inverse frequency of dominant topic in historical set (simple, reproducible)
    counts = hist["dom_topic"].value_counts(dropna=False).to_dict()
    total = sum(counts.values()) if counts else 1
    invfreq = {k: 1.0 - (v / total) for k, v in counts.items()}

    recent_out = recent[["id","title","year","field"]].copy()
    recent_out["B3_topic_pair_rarity"] = recent["dom_topic"].map(lambda z: invfreq.get(z, 1.0))

    out_csv = os.path.join(outdir, "baseline_topic.csv")
    recent_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(recent_out.head())
    print(f"[OK] wrote {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    main(cfg, args.outdir)
