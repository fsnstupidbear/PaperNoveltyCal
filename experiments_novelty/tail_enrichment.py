# tail_enrichment.py
# Evaluate "top-tail enrichment" of recent (novel) papers using thresholds from the historical score distribution.
# Outputs: CSV + MD tables and two figures (lift bar chart; precision/recall/coverage curves).

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_scores(path, score_col, label_col="split"):
    df = pd.read_csv(path)
    # Fallback: infer split from id prefix if 'split' missing
    if label_col not in df.columns:
        if "id" in df.columns:
            df[label_col] = np.where(df["id"].astype(str).str.startswith("R"), "recent", "historical")
        else:
            raise KeyError(f"'{label_col}' column not found and no 'id' column to infer from.")
    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not found in CSV columns: {list(df.columns)}")

    # normalize label values
    df[label_col] = df[label_col].astype(str).str.lower().map(lambda x: "recent" if x.startswith("r") else ("historical" if x.startswith("h") else x))
    # drop NaN scores
    n0 = len(df)
    df = df[~df[score_col].isna()].copy()
    dropped = n0 - len(df)
    return df, dropped

def top_tail_stats(recent_scores, hist_scores, thr):
    # recent=positives; historical=negatives
    tp = int((recent_scores >= thr).sum())
    fp = int((hist_scores   >= thr).sum())
    fn = int((recent_scores <  thr).sum())
    tn = int((hist_scores   <  thr).sum())
    pos = tp + fn
    neg = fp + tn
    total = pos + neg
    base_rate = pos / total if total > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / pos if pos > 0 else 0.0
    coverage = (tp + fp) / total if total > 0 else 0.0
    lift = (precision / base_rate) if (base_rate and base_rate > 0) else np.nan
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pos": pos, "neg": neg, "total": total,
        "base_rate": base_rate,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "lift": lift
    }

def bootstrap_ci_tail(recent_scores, hist_scores, thr, rounds=2000, seed=42):
    """Bootstrap CIs for precision/recall/coverage/lift with fixed threshold thr.
       Resample recent & historical pools separately with replacement."""
    rng = np.random.default_rng(seed)
    n_r = len(recent_scores)
    n_h = len(hist_scores)
    prec, rec, cov, lift = [], [], [], []
    for _ in range(rounds):
        r_idx = rng.integers(0, n_r, n_r)
        h_idx = rng.integers(0, n_h, n_h)
        r_s = recent_scores[r_idx]
        h_s = hist_scores[h_idx]
        st = top_tail_stats(r_s, h_s, thr)
        prec.append(st["precision"])
        rec.append(st["recall"])
        cov.append(st["coverage"])
        lift.append(st["lift"])
    def ci(arr):
        lo, hi = np.quantile(arr, [0.025, 0.975])
        return float(lo), float(hi)
    return ci(prec), ci(rec), ci(cov), ci(lift)

def main(scores_csv, score_col, outdir, quants, label_col, boot, seed):
    ensure_dir(outdir)
    df, dropped = load_scores(scores_csv, score_col, label_col)
    recent = df[df[label_col] == "recent"][score_col].astype(float).values
    hist   = df[df[label_col] == "historical"][score_col].astype(float).values
    if len(recent) == 0 or len(hist) == 0:
        raise ValueError(f"Not enough data: recent={len(recent)}, historical={len(hist)}")

    base_rate = len(recent) / (len(recent) + len(hist))
    rows = []
    for q in sorted(quants):
        # threshold from historical distribution (upper tail)
        thr = float(np.quantile(hist, q))
        stats = top_tail_stats(recent, hist, thr)
        prec_ci, rec_ci, cov_ci, lift_ci = bootstrap_ci_tail(recent, hist, thr, rounds=boot, seed=seed)
        rows.append({
            "quantile": q,
            "hist_threshold": thr,
            "hist_tail_frac": float(1.0 - q),
            "base_rate": float(base_rate),
            "precision": float(stats["precision"]),
            "precision_CI_low": prec_ci[0],
            "precision_CI_high": prec_ci[1],
            "recall": float(stats["recall"]),
            "recall_CI_low": rec_ci[0],
            "recall_CI_high": rec_ci[1],
            "coverage": float(stats["coverage"]),
            "coverage_CI_low": cov_ci[0],
            "coverage_CI_high": cov_ci[1],
            "lift": float(stats["lift"]),
            "lift_CI_low": lift_ci[0],
            "lift_CI_high": lift_ci[1],
            "TP": stats["tp"], "FP": stats["fp"], "FN": stats["fn"], "TN": stats["tn"],
            "recent_N": int(stats["pos"]), "hist_N": int(stats["neg"]), "total_N": int(stats["total"])
        })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, f"TailEnrich_{score_col}.csv")
    out_md  = os.path.join(outdir, f"TailEnrich_{score_col}.md")
    out_json = os.path.join(outdir, f"TailEnrich_{score_col}.json")
    out_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "scores_csv": os.path.basename(scores_csv),
            "score_col": score_col,
            "dropped_rows_due_to_nan": dropped,
            "recent_N": int(len(recent)),
            "historical_N": int(len(hist)),
            "base_rate": float(base_rate),
            "rows": rows
        }, f, ensure_ascii=False, indent=2)

    # Markdown table
    md_lines = []
    md_lines.append(f"# Tail Enrichment (@{score_col})\n")
    md_lines.append(f"- Source: `{os.path.basename(scores_csv)}` | score_col: `{score_col}`")
    md_lines.append(f"- N_recent={len(recent)}, N_historical={len(hist)}, base_rate={base_rate:.4f}, dropped NaN={dropped}\n")
    md_lines.append("| quantile | hist_threshold | tail_frac | precision (95% CI) | recall (95% CI) | coverage (95% CI) | lift (95% CI) | TP | FP | FN | TN |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['quantile']:.3f} | {r['hist_threshold']:.6g} | {r['hist_tail_frac']:.3f} | "
            f"{r['precision']:.3f} [{r['precision_CI_low']:.3f},{r['precision_CI_high']:.3f}] | "
            f"{r['recall']:.3f} [{r['recall_CI_low']:.3f},{r['recall_CI_high']:.3f}] | "
            f"{r['coverage']:.3f} [{r['coverage_CI_low']:.3f},{r['coverage_CI_high']:.3f}] | "
            f"{r['lift']:.2f} [{r['lift_CI_low']:.2f},{r['lift_CI_high']:.2f}] | "
            f"{r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Figure 1: Lift by quantile with CI
    fig1 = os.path.join(outdir, f"Figure_tail_lift_{score_col}.png")
    qs = [r["quantile"] for r in rows]
    lifts = [r["lift"] for r in rows]
    lift_lo = [r["lift_CI_low"] for r in rows]
    lift_hi = [r["lift_CI_high"] for r in rows]
    err_low = np.array(lifts) - np.array(lift_lo)
    err_hi  = np.array(lift_hi) - np.array(lifts)
    plt.figure(figsize=(7,4))
    plt.errorbar(qs, lifts, yerr=[err_low, err_hi], fmt='o-', capsize=4)
    plt.axhline(1.0, linestyle='--')
    plt.xlabel("Historical quantile threshold q")
    plt.ylabel("Lift = Precision / base_rate")
    plt.title(f"Tail enrichment (score={score_col})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    # Figure 2: Precision / Recall / Coverage vs quantile
    fig2 = os.path.join(outdir, f"Figure_tail_pr_rec_{score_col}.png")
    prec = [r["precision"] for r in rows]
    rec  = [r["recall"] for r in rows]
    cov  = [r["coverage"] for r in rows]
    plt.figure(figsize=(7,4))
    plt.plot(qs, prec, marker='o', label="Precision")
    plt.plot(qs, rec, marker='o', label="Recall")
    plt.plot(qs, cov, marker='o', label="Coverage")
    plt.xlabel("Historical quantile threshold q")
    plt.ylabel("Rate")
    plt.title(f"Precision/Recall/Coverage vs q (score={score_col})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_md}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True, help="Path to scores CSV (e.g., out_cum/scores_all.csv)")
    ap.add_argument("--score_col", required=True, help="Column name of the score to evaluate (e.g., s_mtn or S)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--label_col", default="split", help="Label column (default: 'split'); if missing, infer from id prefix R/H")
    ap.add_argument("--quants", nargs="+", type=float, default=[0.95, 0.975, 0.99], help="Historical quantiles to use as thresholds")
    ap.add_argument("--boot", type=int, default=2000, help="Bootstrap rounds for CIs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    main(args.scores_csv, args.score_col, args.outdir, args.quants, args.label_col, args.boot, args.seed)
