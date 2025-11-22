# fuse_rerank_safe.py
import os, json, argparse, math, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def read_jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p,encoding="utf-8") if l.strip()])

def tokenize(s):
    import re
    if not isinstance(s,str): return []
    return [w.lower() for w in re.findall(r"[A-Za-z]+", s)]

def precision_at_k(labels,k=10):
    k=min(k,len(labels));
    return float(np.sum(labels[:k]))/float(k) if k>0 else 0.0

def dcg_at_k(labels,k=10):
    k=min(k,len(labels));
    if k==0: return 0.0
    labels=np.asarray(labels[:k],float)
    disc=1.0/np.log2(np.arange(2,k+2))
    return float(np.sum(labels*disc))

def ndcg_at_k(labels,k=10):
    dcg=dcg_at_k(labels,k); ideal=dcg_at_k(sorted(labels,reverse=True),k)
    return 0.0 if ideal<=0 else float(dcg/ideal)

def main(outdir, scores_csv, topN=300, r_percent=20.0, include_self=True):
    pool = read_jsonl(os.path.join(outdir,"pool_ALL.jsonl"))
    S = pd.read_csv(scores_csv)[["id","S"]]
    B4 = pd.read_csv(os.path.join(outdir,"baseline_graph.csv"))[["id","B4_graph_newedge"]]
    df = pool.merge(S,on="id",how="left").merge(B4,on="id",how="left")
    recent = df[df["split"]=="recent"].copy().reset_index(drop=True)
    Q=len(recent);
    if Q==0: raise RuntimeError("No recent docs")
    # TF-IDF on recent
    vec=TfidfVectorizer(analyzer=tokenize, max_features=40000, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    X=vec.fit_transform(recent["text"].astype(str).tolist()); sim=(X@X.T).toarray()  # cosine

    # 标签：池内 top-r% newedge 为正例 (+self)
    grid=[0.2,0.4,0.6,0.8]
    out=[]
    for lam in grid:
        P, N, valid= [], [], 0
        for i in range(Q):
            order=np.argsort(-sim[i])
            if not include_self: order=order[order!=i]
            top_idx=order[:topN] if len(order)>topN else order
            cand_S = recent.iloc[top_idx]["S"].values
            cand_ne= recent.iloc[top_idx]["B4_graph_newedge"].fillna(0).values
            # 归一化 cosine
            c = sim[i, top_idx]
            c = (c - c.min()) / (c.max()-c.min() + 1e-12)
            score = lam*cand_S + (1-lam)*c

            k_pos=max(1,int(math.floor(len(top_idx)*r_percent/100.0)))
            thr = np.sort(cand_ne)[::-1][k_pos-1] if len(top_idx)>=k_pos else np.max(cand_ne)
            labels = (cand_ne>=thr).astype(int)
            if include_self:
                self_pos = np.where(top_idx==i)[0]
                if len(self_pos)>0: labels[self_pos[0]]=1
            if labels.sum()==0:
                continue
            ord2=np.argsort(-score)
            ranked=labels[ord2]
            P.append(precision_at_k(ranked,10)); N.append(ndcg_at_k(ranked,10)); valid+=1
        meta={"lam":lam, "P@10":float(np.mean(P)) if P else 0.0, "nDCG@10":float(np.mean(N)) if N else 0.0, "Q":valid,
              "topN":topN, "r%":r_percent, "self":include_self}
        out.append(meta); print(meta)
    # 选择最优
    best=max(out, key=lambda d:(d["nDCG@10"], d["P@10"]))
    with open(os.path.join(outdir,"fuse_grid.json"),"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
    with open(os.path.join(outdir,"fuse_best.json"),"w",encoding="utf-8") as f: json.dump(best,f,ensure_ascii=False,indent=2)
    print("[Best]", best)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--topN", type=int, default=300)
    ap.add_argument("--r_percent", type=float, default=20.0)
    ap.add_argument("--include_self", action="store_true", default=True)
    args=ap.parse_args()
    main(args.outdir, args.scores_csv, args.topN, args.r_percent, args.include_self)
