# baselines_graph.py — Graph rarity baseline (PMI & new-edge) using only `text` (title+abstract).
# Works with the cumulative-score pipeline; no `methods/tasks` columns required.

import os, json, argparse, math, re
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
    raise RuntimeError("scikit-learn is required. Please `pip install scikit-learn`.") from e

try:
    import yaml
except Exception:
    yaml = None

def load_cfg(path):
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Please `pip install pyyaml`.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Simple English tokenizer; you can adjust to your corpus.
_TOKEN = re.compile(r"[A-Za-z]+")

_DEFAULT_STOPS = set("""
a an the and or of for to in on with without within via by from this that these those into as at is are was were be been being
we you he she they it its their our my your his her them which who whose what when where why how
using use based method task approach model models paper study result results new novel
""".split())

def simple_tokenize(s: str):
    if not isinstance(s, str):
        return []
    toks = [w.lower() for w in _TOKEN.findall(s)]
    return [w for w in toks if w not in _DEFAULT_STOPS and len(w) >= 3]

def build_topk_terms(texts, vocab_hist, topk=12, fit_vectorizer=True, max_features=20000):
    """
    Fit TF-IDF on *historical texts* to avoid leakage, then select top-k terms per doc.
    texts: list[str]
    vocab_hist: list[str] used to fit; if None -> fit on 'texts'
    """
    analyzer = simple_tokenize
    vec = TfidfVectorizer(
        analyzer=analyzer,
        max_features=max_features,
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    # fit on historical vocabulary (list of strings)
    vec.fit(vocab_hist if vocab_hist is not None else texts)

    X = vec.transform(texts)  # csr
    idx2term = np.array(vec.get_feature_names_out())

    topk_terms_per_doc = []
    for i in range(X.shape[0]):
        row = X[i]
        if row.nnz == 0:
            topk_terms_per_doc.append([])
            continue
        data = row.data
        inds = row.indices
        if len(data) <= topk:
            terms = idx2term[inds].tolist()
        else:
            top_idx = np.argpartition(-data, topk-1)[:topk]
            terms = idx2term[inds[top_idx]].tolist()
        topk_terms_per_doc.append(sorted(set(terms)))
    return topk_terms_per_doc

def pmi_stats(hist_term_sets, n_hist_docs, eps=1e-9):
    """
    Compute doc-level PMI on historical term sets.
    n_i  : #docs containing term i
    n_ij : #docs containing both (i,j)
    PMI(i,j) = log( (n_ij * N) / (n_i * n_j) )
    """
    term_df = defaultdict(int)
    pair_df = defaultdict(int)

    for terms in hist_term_sets:
        if not terms or len(terms) < 2:
            # count singletons anyway
            for t in set(terms):
                term_df[t] += 1
            continue
        uniq = set(terms)
        for t in uniq:
            term_df[t] += 1
        for a,b in combinations(sorted(uniq), 2):
            pair_df[(a,b)] += 1

    # precompute PMI and its min/max for normalization
    pmi_map = {}
    pmi_min, pmi_max = None, None
    for (a,b), nij in pair_df.items():
        ni, nj = term_df[a], term_df[b]
        # guard
        if ni <= 0 or nj <= 0 or nij <= 0:
            continue
        pmi = math.log((nij * n_hist_docs) / (ni * nj) + eps)
        pmi_map[(a,b)] = pmi
        if pmi_min is None or pmi < pmi_min:
            pmi_min = pmi
        if pmi_max is None or pmi > pmi_max:
            pmi_max = pmi

    # if no pairs found, set safe defaults
    if pmi_min is None:
        pmi_min, pmi_max = 0.0, 1.0

    return term_df, pair_df, pmi_map, pmi_min, pmi_max

def score_recent(recent_term_sets, term_df, pair_df, pmi_map, pmi_min, pmi_max):
    """
    For each recent doc:
      - B4_graph_newedge : (#pairs unseen in historical) / (#pairs)
      - B4_graph_pmi_inv : average of (1 - PMI_norm) across its pairs;
                           unseen pairs treated as PMI_norm=0 -> rarity=1
    """
    out_newedge = []
    out_pmiinv = []
    rng = pmi_max - pmi_min if (pmi_max - pmi_min) > 1e-12 else 1.0

    for terms in recent_term_sets:
        if not terms or len(terms) < 2:
            out_newedge.append(0.0)
            out_pmiinv.append(0.5)  # neutral
            continue
        uniq = sorted(set(terms))
        pairs = list(combinations(uniq, 2))
        total = len(pairs)
        unseen = 0
        acc_rarity = 0.0
        for a,b in pairs:
            key = (a,b) if (a,b) in pmi_map else ((b,a) if (b,a) in pmi_map else None)
            if key is None:
                unseen += 1
                acc_rarity += 1.0  # PMI_norm=0 -> rarity=1
            else:
                pmi = pmi_map[key]
                pmi_norm = (pmi - pmi_min) / rng
                rarity = 1.0 - max(0.0, min(1.0, pmi_norm))
                acc_rarity += rarity
        out_newedge.append(unseen / total if total > 0 else 0.0)
        out_pmiinv.append(acc_rarity / total if total > 0 else 0.5)
    return out_pmiinv, out_newedge

def main(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    pool_path = os.path.join(outdir, "pool_ALL.jsonl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"{pool_path} not found. Run data_prep.py first.")

    # load pool
    rows = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    df = pd.DataFrame(rows)
    if "text" not in df.columns or "split" not in df.columns:
        raise KeyError("pool_ALL.jsonl must contain `text` and `split` columns.")

    hist = df[df["split"]=="historical"].copy()
    recent = df[df["split"]=="recent"].copy()

    # Graph config (safe defaults)
    graph_cfg = cfg.get("graph", {}) if isinstance(cfg.get("graph", {}), dict) else {}
    topk_terms    = int(graph_cfg.get("topk_terms", 12))
    max_features  = int(graph_cfg.get("max_features", 20000))

    # Prepare texts
    hist_texts   = hist["text"].astype(str).tolist()
    recent_texts = recent["text"].astype(str).tolist()

    # Fit TF-IDF on historical texts only; select top-k terms per doc
    hist_term_sets   = build_topk_terms(hist_texts, vocab_hist=hist_texts, topk=topk_terms, max_features=max_features)
    recent_term_sets = build_topk_terms(recent_texts, vocab_hist=hist_texts, topk=topk_terms, max_features=max_features)

    # Build historical PMI stats
    term_df, pair_df, pmi_map, pmi_min, pmi_max = pmi_stats(hist_term_sets, n_hist_docs=len(hist_texts))

    # Score recent docs
    pmi_inv, newedge = score_recent(recent_term_sets, term_df, pair_df, pmi_map, pmi_min, pmi_max)

    recent_out = recent[["id","title","year","field"]].copy()
    recent_out["B4_graph_pmi_inv"] = pmi_inv
    recent_out["B4_graph_newedge"] = newedge

    out_csv = os.path.join(outdir, "baseline_graph.csv")
    recent_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(recent_out.head())
    print(f"[OK] wrote {out_csv}  | hist_docs={len(hist_texts)} recent_docs={len(recent_texts)} pairs={len(pmi_map)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    main(cfg, args.outdir)
