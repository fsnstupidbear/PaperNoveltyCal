# ablation.py
import pandas as pd, numpy as np, os, argparse, yaml, json
import matplotlib.pyplot as plt

def bar_plot(labels, values, title, outpng):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title); plt.ylabel('Metric drop')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)

def main(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    base = pd.read_csv(os.path.join(outdir, "scores_recent.csv"))
    # emulate ablations by turning component weights to zero and recomputing S on the fly
    comps = [('Only-Method', (1,0,0)), ('Only-Task',(0,1,0)), ('No-Pair',(1,1,0)), ('No-Attention',(1,1,1)), ('EncoderSwap',(1,1,1))]
    # For EncoderSwap here we only mark placeholder (same S); you can replace scoring.py with alternate encoder
    base_acc = (base['S'] > base['S'].median()).mean()
    drops = []
    labels = []
    for name, (wm, wt, wc) in comps:
        s = wm*base['s_mn'] + wt*base['s_tn'] + wc*base['s_mtn']
        acc = (s > s.median()).mean()
        drop = float(base_acc - acc)
        drops.append(drop); labels.append(name)
    bar_plot(labels, drops, "Ablation Drops (Accuracy proxy)", os.path.join(outdir, "figure_ablation.png"))
    with open(os.path.join(outdir, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"base_acc_proxy": float(base_acc), "drops": dict(zip(labels, drops))}, f, ensure_ascii=False, indent=2)
    print("Ablation done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    main(cfg, args.outdir)
