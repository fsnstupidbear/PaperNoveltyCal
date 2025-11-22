# PaperNoveltyCal

A lightweight, non-LLM framework for **entity-centric novelty scoring of scientific papers**.  
The system extracts *method* and *task* entities from titles / abstracts, builds explicit **(method, task)** pairs, and assigns a document-level novelty score based on how rare these combinations are in a historical corpus.

This repository contains the code used in our paper on method–task pair based novelty assessment.

---

## 1. Main ideas

- Move from **document-level similarity** to **entity- and relation-level novelty**.
- Automatically extract **method** and **task** entities and their **used-for** relations.
- Build a historical knowledge base of entities and (method, task) pairs.
- Compute three novelty components for each paper  

  - `s_m` – method-level novelty  
  - `s_t` – task-level novelty  
  - `s_m×t` (`s_mxt`) – method–task pair novelty  

- Aggregate the components with attention-style weights into a final document score `S`.
- Show that the pair score `s_m×t` alone is already effective at separating **high- vs. low-novelty** papers without using publication year as an input feature.

---

## 2. Repository layout (high level)

The exact folders may vary slightly, but the core code is:

- `experiments_novelty/` – all scoring & evaluation scripts  
  - `config.cumulative.yaml` – example configuration for historical + recent papers  
  - `data_prep.py` – normalize raw CSVs and build historical / recent pools  
  - `scoring.py` – compute `s_m`, `s_t`, `s_m×t` and final score `S`  
  - `eval_binary_global.py`, `eval_binary_plus.py` – global binary classification & PR evaluation  
  - `eval_pairwise_pack.py`, `eval_retrieval_v3.py` – pairwise and retrieval-style evaluation  
  - `run_s_mtn_pack.py` – convenience script to reproduce the main `s_m×t` experiments  
  - other utilities for baselines, plots, and ablation tables

You can inspect each script for more details about inputs and outputs.

Raw CSV data files with pre-extracted entities and scores are **not** included in the repository; please prepare your own data and update the paths in the YAML config accordingly.

---

## 3. Environment

The code is implemented in Python and uses only standard ML / data-science libraries.

Typical environment (one example):

- Python 3.9+  
- `numpy`, `pandas`, `scikit-learn`, `scipy`  
- `PyYAML`, `tqdm`, `matplotlib`  
- `gensim` or `sentence-transformers` for text embeddings (depending on your setup)

You can either:

```bash
pip install -r requirements.txt
