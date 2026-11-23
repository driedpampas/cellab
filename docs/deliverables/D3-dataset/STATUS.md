# Deliverable 3: what is done and what is left

Written 23 November 2026. Deliverable 3 (Lab 11) asks for a **dataset of generations** and a **neural-network
(or reinforcement-learning) experiment** on it. How the experiment is designed and why is in
[README.md](README.md); this page is the checklist.

> **Next step:** run the v2 notebook once (see "3b" below). The results in `results/` come from two code
> versions, so the shared-vs-separate comparison is not fair yet, and the exact simulated answers are not in yet.

## Summary

| Part | State |
|---|---|
| All 15 automata from the brief | Done |
| Dataset generator | Done, tested |
| The datasets (18 of them, 94,380 runs) | Done, in `data/` |
| Question/answer set for hand-in | Done for the elementary rules; one command for the rest |
| The two networks, training, evaluation & verification code | Done, tested on Linux (`299 passed`) |
| Colab notebook | **Done & verified on L4 GPU (30.8 min total)** |
| One-step network on the 256 elementary rules | **Done — 100.0% seen, unseen, 32-cell & live engine** |
| Five-questions network (shared + 5 separate) | Trained; **comparison provisional** (the two sides used different code, one seed each) — v2 run pending |
| All five answers by simulating with the learned step (`ml/simulate.py`) | Code done and checked locally (exact on 650 class-3/4 runs incl. Rule 30); **full run in the v2 notebook** |
| 2D automata + traffic training | **Done — 8/10 discrete deterministic StepNets at 100.0% N=32 + `diffusion_16` float support (MAE 0.52)** |
| Gradient-boosting baseline (`HistGradientBoosting`) | Done — `AnswerNet` is better on unseen count, stable (F1 0.600 vs 0.365) and occurs (0.741 vs 0.661); neither has skill on chaotic cycle lengths |
| Post-training verification & speed benchmarks | Done. The checks are now stricter: a model no better than the baseline is a WARN (Brian's Brain, Game of Life and predator-prey five-questions networks are exactly at the baseline) |
| Presentation for Deliverable 3 | **Not started** (ready — all results & charts archived) |

## Done

### Automata (needed before the dataset)
- The **agents mode** (`src/cellab/agents.py`): a rule can change one other cell, cells can carry extra
  numbers ("layers") such as energy or heading, and each automaton chooses the order in which cells act.
- The 5 automata that needed it: `langtons_ant`, `falling_sand`, `schelling`, `predator_prey`, `ecosystem`,
  each with an experiment file. Langton's ant and falling sand are checked step by step against
  independent implementations.
- Fingerprints of the older automata are unchanged, so every number on the Deliverable 2 slides still holds
  (a test checks this).

### Dataset
- `cellab dataset datasets/<name>.json` generates a dataset; `cellab qa data/<name>.npz` writes it as
  questions and answers in words. Format: [docs/FILE_FORMATS.md](../../FILE_FORMATS.md#dataset-file).
- 18 datasets in `data/`, listed with their statistics in [docs/DATASETS.md](../../DATASETS.md):
  - `elementary_w16`: all 256 elementary rules, 200 runs each, 16 cells. Every run is followed until its
    cycle is found, so all five answers are exact. It reproduces the Deliverable 1 numbers (Rule 30's cycle of 4,144).
  - `elementary_w32_test`: the same rules on 32 cells, to test other grid sizes.
  - One dataset for each of the other 16 automata (16 x 16 grids, 1,000 to 3,000 runs each).
- Every sample has its own seed, so the data is identical each time it is generated.
- `data/elementary_w16_qa.jsonl.gz`: 1,280,000 questions and answers in words (9 MB), for the hand-in.
  `data/qa_sample.jsonl` is a small readable sample.
- Tests check that every label equals the answer of an ordinary `cellab run` (`tests/test_dataset.py`).

### Networks and training code (`ml/`)
- `ml/eca.py`: the 88 families of equivalent rules and Wolfram's 4 classes. 15% of the families of each
  class are never shown in training (Rule 30's family is among them), to test rules the network never saw.
- `ml/models.py`: two networks.
  - **StepNet** learns one step and is applied N times.
  - **AnswerNet** answers all five questions in one pass, with a shared encoder and five heads.
- `ml/train.py`: training with resume after a disconnect. `--tasks` trains single-question networks, for
  the "one network or five?" comparison from the lab notes.
- `ml/evaluate.py`: the metrics announced in Deliverable 1, by Wolfram class and by steps ahead, with
  baselines (copy the start; the most common answer).
- `notebooks/D3_train_on_colab.ipynb`: runs all of the training and evaluation, saving to Google Drive.
- Tests (`tests/test_ml.py`, 35 of the 293 tests; they need PyTorch).

### First result
The one-step network (10,434 parameters, 26 seconds on a laptop CPU) gets **every grid completely right
64 steps ahead**, also for the 7,600 runs from rule families it never saw, and also on rows of 32 cells
after training on 16. Numbers and chart: [results/step_elementary/](results/step_elementary/metrics.json).

A quick 10-epoch check on Brian's Brain (2D) reached 100% of cells after 1 step and 88–90% after 16–32
steps, so it did not learn the step exactly in that short run. Full training is on the to-do list.

## To do

### 1. Push the work to GitHub
- [x] Pushed to `origin/main` (including Colab optimizations, verification suite, and full D3 results).

### 2. Set up and check on Linux (Matei)
- [x] Venv created, dependencies installed, all **295 tests pass** on Linux.

### 3. Run the training on Colab (Matei)
- [x] Optimized `notebooks/D3_train_on_colab.ipynb` + `ml/` for Colab Pro **L4 GPU** (`bf16`, fused AdamW, fused circular `F.pad`, local SSD training + Drive sync, automatic per-model and post-training verification & speed benchmarks).
- [x] Ran all 34 models end-to-end on L4 (**30.8 minutes** total; `PASS: 47 | WARN: 1 | FAIL: 0`).
- [x] Downloaded `cellab_results/` into local `results/`, archived all 34 models' `metrics.json`, `accuracy_by_horizon.png`, `summary.csv`, `verification.json`, `verification_report.md`, `cpu_speed_benchmark.json`, `multitask_vs_single.png`, and `speed_benchmark.png` into `docs/deliverables/D3-dataset/results/`.

### 3b. The v2 run on Colab (Matei)
- [ ] Open `notebooks/D3_train_on_colab.ipynb` in Colab, **L4 GPU**, *Runtime → Run all*. It writes to the new Drive
      folder `cellab_results_v2`, so everything is retrained on the current code. About an hour with
      `SEEDS = (0, 1, 2)`; set `SEEDS = (0,)` to halve it. Seed 0 of everything runs first, extra seeds last.
      The runtime disconnects itself at the end, also after an error in one model.
- [ ] Copy `metrics.json` / plots / `verification_report.md` from `cellab_results_v2/` into `docs/deliverables/D3-dataset/results/`
      (replace the old ones) and update the provisional sections of `README.md` (2b, 3).

What to look at in `verification_report.md`:
- **"StepNet + simulation: exact whole-run answers"** should be PASS for test and unseen (cycles exact, Rule 30 included).
- **"Cycle length by Wolfram class"**: AnswerNet good on classes 1–2, no skill on 3–4; simulation exact everywhere.
- **"Shared vs Separate"** with mean ± sd over 3 seeds; "within noise" means do not claim a winner.

### 4. Hand-in material
- [ ] Question/answer sets for the other datasets, if the hand-in should include them (about 1 minute each):
      `cellab qa data/<name>.npz`.
- [x] Updated `docs/deliverables/D3-dataset/README.md` with the real results, failure-mode analysis, and GPU/CPU benchmarks.

### 5. Presentation for Lab 11 (both)
Same style as the Deliverable 2 deck: plain, descriptive titles; real numbers only.
- [ ] What was asked and our answer (dataset + experiment).
- [ ] The dataset: how it is made, sizes, splits, why the labels are exact.
- [ ] Encoding and the two networks (with the "one step = a 3-wide convolution" link to the engine's shifted views).
- [ ] Results: one-step network (exact, also on unseen rules); five-questions network by class and N;
      one network vs five; baselines.
- [ ] What it means: a network can learn the rules exactly, but skipping the simulation fails where the
      automaton is chaotic.
- [ ] Live demo: `cellab dataset` on a small file, `cellab qa`, and the evaluation of a trained model.
- [ ] Short mention of the new agents-mode automata (all 15 from the brief now exist).

### 6. Optional extras, by value
- [x] A gradient-boosting baseline (`ml/baselines.py`, `HistGradientBoosting`) for counts and whole-run questions.
- [ ] Asking a language model the same questions directly, as the "why we simulate" comparison.
- [x] Support for decimal values (`diffusion_16` trained & evaluated via L1 regression).
- [ ] A reinforcement-learning agent that edits the starting grid towards a goal. The deliverable says
      "neural network **or** reinforcement learning", so this is not required.

### Reminders for both
- Every conversation with an AI assistant must be saved (course rule). Claude Code sessions in this folder
  are saved automatically; other assistants: export into `docs/ai-logs/raw/`.
- Commit your own work under your own name: individual contribution can change the final grade.
