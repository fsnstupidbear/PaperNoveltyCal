# baselines_lexical.py
import pandas as pd, numpy as np, os, yaml, argparse
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(s: str):
    return s.lower().split()

def tfidf_nn_distance(q_vec, D):
    sims = cosine_similarity(q_vec, D)
    return 1.0 - float(sims.max()) if sims.size>0 else 1.0

def main(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    hist = pd.read_json(os.path.join(outdir, "historical.norm.jsonl"), lines=True)
    rec  = pd.read_json(os.path.join(outdir, "recent.norm.jsonl"), lines=True)

    # TF-IDF cosine on title+abstract
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_hist = vec.fit_transform((hist['title'] + ' ' + hist['abstract']).str.lower())
    X_rec  = vec.transform((rec['title'] + ' ' + rec['abstract']).str.lower())

    tfidf_dist = []
    for i in range(X_rec.shape[0]):
        tfidf_dist.append(tfidf_nn_distance(X_rec[i], X_hist))
    rec_out = rec[['id','title','year','field']].copy()
    rec_out['B1_tfidf_dist'] = tfidf_dist

    # BM25 on title+abstract
    corpus = [tokenize(t) for t in (hist['title'] + ' ' + hist['abstract'])]
    bm25 = BM25Okapi(corpus)
    bms = []
    for txt in (rec['title'] + ' ' + rec['abstract']):
        scores = bm25.get_scores(tokenize(txt))
        bms.append(-float(np.max(scores)) if len(scores)>0 else 0.0) # more negative = rarer
    rec_out['B1_bm25_negmax'] = bms

    rec_out.to_csv(os.path.join(outdir, "baseline_lexical.csv"), index=False)
    print(rec_out.head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    main(cfg, args.outdir)
