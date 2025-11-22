# run_s_mtn_pack.py
import subprocess, sys
def run(cmd):
    print(">>", " ".join(cmd)); subprocess.run(cmd, check=True)
outdir = "out_cum/"
run([sys.executable, "eval_binary_global.py", "--outdir", outdir, "--scores_csv", outdir+"scores_all.csv", "--score_col", "s_mtn"])
run([sys.executable, "make_table_significance.py", "--in_csv", outdir+"scores_all.csv", "--outdir", outdir])
run([sys.executable, "decile_lift.py", "--in_csv", outdir+"scores_all.csv", "--outdir", outdir, "--col", "s_mtn"])
run([sys.executable, "logistic_s_mtn.py", "--in_csv", outdir+"scores_all.csv", "--outdir", outdir, "--col", "s_mtn"])
run([sys.executable, "plot_distributions.py", "--in_csv", outdir+"scores_all.csv", "--outdir", outdir, "--col", "s_mtn"])
print("\n[ALL DONE] Artifacts are in", outdir)
