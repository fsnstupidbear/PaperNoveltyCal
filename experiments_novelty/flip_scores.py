# -*- coding: utf-8 -*-
import argparse, os, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--flip", action="store_true")
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
if "S" not in df.columns:
    raise SystemExit("Input CSV must have column 'S'.")

out = df.copy()
if args.flip:
    out["S"] = -out["S"]
out.to_csv(args.out_csv, index=False, encoding="utf-8")
print(f"[OK] wrote {args.out_csv}, flip={args.flip}")
