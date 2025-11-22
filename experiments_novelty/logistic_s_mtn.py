# logistic_s_mtn.py
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def fit_sm(X, y):
    try:
        import statsmodels.api as sm
        X1 = sm.add_constant(X)
        model = sm.Logit(y, X1).fit(disp=False)
        coef = model.params[1]; se = model.bse[1]
        z = 1.96
        lo, hi = coef - z*se, coef + z*se
        p = model.pvalues[1]
        return dict(coef=float(coef), OR=float(np.exp(coef)),
                    OR_CI=(float(np.exp(lo)), float(np.exp(hi))), p=float(p),
                    ok=True, yhat=model.predict(X1).values.tolist())
    except Exception as e:
        # bootstrap 近似
        rng = np.random.default_rng(42)
        coefs=[]
        for _ in range(500):
            idx=rng.integers(0,len(y),len(y))
            xb=X[idx]; yb=y[idx]
            # 最小二乘近似 logit -> 简化（不输出 p 值）
            from sklearn.linear_model import LogisticRegression
            lr=LogisticRegression(max_iter=1000).fit(xb.reshape(-1,1), yb)
            coefs.append(lr.coef_[0,0])
        coef=np.mean(coefs); se=np.std(coefs)+1e-9
        lo, hi = coef-1.96*se, coef+1.96*se
        from sklearn.linear_model import LogisticRegression
        lr=LogisticRegression(max_iter=1000).fit(X.reshape(-1,1), y)
        yhat = lr.predict_proba(X.reshape(-1,1))[:,1]
        return dict(coef=float(coef), OR=float(np.exp(coef)),
                    OR_CI=(float(np.exp(lo)), float(np.exp(hi))), p=None,
                    ok=False, yhat=yhat.tolist())

def main(in_csv, outdir, col):
    os.makedirs(outdir, exist_ok=True)
    df=pd.read_csv(in_csv)[[col,"split"]].dropna().copy()
    y=(df["split"]=="recent").astype(int).values
    X=df[col].values.reshape(-1,1)
    pack=fit_sm(X, y)

    with open(os.path.join(outdir, f"logit_{col}.txt"),"w",encoding="utf-8") as f:
        f.write(str(pack))

    prob=np.array(pack["yhat"])
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5,5))
    plt.plot(mean_pred, frac_pos, 'o-')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Predicted prob")
    plt.ylabel("Observed freq (Recent)")
    plt.title(f"Calibration by {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"Figure_calibration_{col}.png"), dpi=200)
    print("[OK] wrote logit report & calibration figure for", col)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="out_cum/scores_all.csv")
    ap.add_argument("--outdir", default="out_cum/")
    ap.add_argument("--col", default="s_mtn")
    args=ap.parse_args()
    main(args.in_csv, args.outdir, args.col)
