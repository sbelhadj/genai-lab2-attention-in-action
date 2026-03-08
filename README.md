# Lab 2 — Attention in Action

**Generative AI & Prompt Engineering — A Mechanistic Approach**

Module 2: Transformer & Attention | Duration: 90 minutes

---

## Overview

In this lab, you will **build the attention mechanism from scratch** in NumPy, then run empirical experiments showing how context shapes a real LLM's output. You will discover that attention is the mechanism by which your prompt actually influences generation — and this insight explains why prompt structure, few-shot examples, and context injection work.

**Core principle:** *Attention is how the model decides which parts of your prompt matter for generating each output token. It is not magic — it is weighted averaging over context.*

---

## Quick Start

### Option A: GitHub Codespaces (Recommended)

1. Click **"Code"** → **"Codespaces"** → **"Create codespace on main"**
2. Wait for the environment to build (~3–5 minutes)
3. Open `lab2_attention_in_action.ipynb`

### Option B: Local Setup

```bash
git clone <your-repo-url>
cd genai-lab2-attention-in-action
pip install numpy matplotlib jupyter requests pytest nbformat
# For Part B only:
ollama serve &
ollama pull llama3.2:3b
jupyter notebook
```

### Verify Setup

```bash
python -c "import numpy as np; print(f'✓ NumPy {np.__version__}')"
python -c "import matplotlib; print('✓ Matplotlib ready')"
curl -s http://localhost:11434/api/tags 2>/dev/null && echo "✓ Ollama running" || echo "⚠ Ollama not running (Part A still works)"
```

---

## Repository Structure

```
genai-lab2-attention-in-action/
├── .devcontainer/devcontainer.json       # Codespaces config
├── .github/workflows/autograding.yml     # CI auto-grading
├── lab2_attention_in_action.ipynb        # ← YOUR MAIN WORKSPACE
├── utils/
│   ├── __init__.py
│   ├── attention_utils.py                # Provided: softmax(), plot_attention()
│   ├── attention_impl.py                 # YOUR CODE: paste implementations here
│   └── generation_utils.py               # Provided: generate(), compare_outputs()
├── data/
│   └── precomputed_outputs.json          # Fallback if Ollama unavailable
├── tests/
│   ├── __init__.py
│   ├── test_attention.py                 # Auto-graded: attention implementations
│   └── test_notebook_structure.py        # Auto-graded: notebook completeness
├── README.md
└── .gitignore
```

---

## What to Do

1. Open `lab2_attention_in_action.ipynb`
2. **Part A (45 min):** Implement attention in NumPy — 5 exercises
3. **Part B (35 min):** Run 3 experiments with real model outputs
4. **Copy your implementations** to `utils/attention_impl.py` for auto-grading
5. Commit and push

---

## Deliverables

| # | What | Where |
|---|------|-------|
| 1 | Completed notebook | `lab2_attention_in_action.ipynb` |
| 2 | Attention implementations | `utils/attention_impl.py` |
| 3 | All heatmap visualizations | In notebook |
| 4 | 5+ written observations | In notebook |
| 5 | 3 experiments with analysis | In notebook |
| 6 | Mini-report (1 page) | In notebook |

---

## Submitting

```bash
git add -A
git commit -m "Lab 2 complete"
git push
```

Check **Actions** tab for auto-grading results.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Part A works but Part B needs Ollama | Use `data/precomputed_outputs.json` — see fallback cells in notebook |
| Heatmap labels overlap | Increase figure size: `figsize=(12, 8)` |
| `scaled_dot_product_attention` gives wrong shapes | Check: Q is (seq_len, d_k), K is (seq_len, d_k), V is (seq_len, d_v) |
| Auto-grading fails | Copy your 3 functions to `utils/attention_impl.py` with exact function names |

---

## Prerequisites from Lab 1

You should know: tokens ≠ words, logits → softmax → probabilities, temperature/top-k/top-p. Today we open the Transformer box to see **how** those logits are computed.

---

*Lab 2 of 8 — DevAssist / TaskFlow Lab Series*
