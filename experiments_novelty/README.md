# Experiments for Novelty Scoring (Independent Add-on)

This folder provides **drop-in, stand-alone** scripts to reproduce the experiments we outlined for EI/SCI 4th-quartile readiness **without modifying your existing code**.

## What’s included
- `config.yaml` — central config (paths, column mapping, weights, time splits)
- `data_prep.py` — unify inputs; build candidate pools
- `scoring.py` — compute our novelty score **S** (entity rarity + task/method combo rarity, min–max normalized)
- `baselines_lexical.py` — TF‑IDF / BM25 baselines (no LLM)
- `baselines_embed.py` — optional SciBERT/E5 embedding distance baseline (graceful fallback if not installed)
- `baselines_topic.py` — LDA topic‑pair rarity baseline
- `baselines_graph.py` — PMI / new‑edge‑rate graph rarity baseline for (method, task)
- `eval_pairwise.py` — pairwise judging metrics (Accuracy, ROC‑AUC, bootstrap CI) + ROC plot
- `eval_retrieval.py` — novelty‑aware retrieval metrics (nDCG@10, Precision@10) + CI
- `ablation.py` — ablations: drop components / swap encoder; bar chart of drops
- `time_robustness.py` — rolling time windows; line plot
- `plot_utils.py` — matplotlib plotting helpers (no seaborn)
- `make_table_main.py` — compose **Table 4** (main results) as CSV/Markdown
- `run_all.py` — one‑click pipeline orchestrator
- `requirements_extras.txt` — extra pip deps (you can install in a fresh venv)
- `examples/` — tiny schema examples for quick dry‑run

## Minimal data expectation
We expect two JSONL files (or CSV) providing paper metadata for two splits:
- `historical_path`: 2016–2022
- `recent_path`: 2023–2025

Columns are configurable; default schema:
```
id, title, abstract, year, field, methods (list|;‑sep), tasks (list|;‑sep)
```
If you **already have entity/combination outputs**, point to those columns in `config.yaml`.

## Quick start
```bash
# (optional) python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements_extras.txt

# edit config.yaml to point to your data
python run_all.py --config config.yaml --outdir out/

# Individual steps:
python eval_pairwise.py   --config config.yaml --outdir out/
python eval_retrieval.py  --config config.yaml --outdir out/
python ablation.py        --config config.yaml --outdir out/
python time_robustness.py --config config.yaml --outdir out/
python make_table_main.py --config config.yaml --outdir out/
```
Plots and tables will be saved in `out/`.
