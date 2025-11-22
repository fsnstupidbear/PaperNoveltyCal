# time_robustness.py
import pandas as pd, numpy as np, os, argparse, yaml, json
import matplotlib.pyplot as plt

def line_plot(xs, ys, title, outpng, xlabel="Split", ylabel="Accuracy proxy"):
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(outpng, dpi=200)

def main(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    splits = cfg.get('time_splits', [])
    ys = []; xs = []
    # We reuse the already-computed recent scores as a placeholder; for real runs, rerun scoring per split.
    base = pd.read_csv(os.path.join(outdir, "scores_recent.csv"))
    acc = (base['S'] > base['S'].median()).mean()
    for i, sp in enumerate(splits):
        xs.append(f"{sp['historical'][0]}-{sp['historical'][1]} / {sp['recent'][0]}-{sp['recent'][1]}")
        ys.append(float(acc))  # placeholder (constant); re-run per split for real curves
    line_plot(xs, ys, "Time Robustness (proxy)", os.path.join(outdir, "figure_time.png"))
    with open(os.path.join(outdir, "time_robustness_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"splits": xs, "acc_proxy": ys}, f, ensure_ascii=False, indent=2)
    print("Time robustness done (proxy).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    main(cfg, args.outdir)
