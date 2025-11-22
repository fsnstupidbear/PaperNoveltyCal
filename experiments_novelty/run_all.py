# run_all.py
import os, argparse, yaml, subprocess, sys

def run(cmd):
    print(">>", " ".join(cmd)); r = subprocess.run(cmd, check=True)
    return r.returncode

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    steps = [
        ["python","data_prep.py","--config",args.config,"--outdir",args.outdir],
        ["python","scoring.py","--config",args.config,"--outdir",args.outdir],
        ["python","baselines_lexical.py","--config",args.config,"--outdir",args.outdir],
        ["python","baselines_topic.py","--config",args.config,"--outdir",args.outdir],
        ["python","baselines_graph.py","--config",args.config,"--outdir",args.outdir],
        ["python","eval_pairwise.py","--config",args.config,"--outdir",args.outdir],
        ["python","eval_retrieval.py","--config",args.config,"--outdir",args.outdir],
        ["python","ablation.py","--config",args.config,"--outdir",args.outdir],
        ["python","time_robustness.py","--config",args.config,"--outdir",args.outdir],
        ["python","make_table_main.py","--config",args.config,"--outdir",args.outdir],
    ]
    for cmd in steps:
        run(cmd)
    print("All steps completed. Outputs in", args.outdir)
