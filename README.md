# CelLab

AI course project: describe cellular automata in files, simulate them, and answer questions about their evolution.

Team: Alexandru Bianca Ana-Maria, Gurzu Matei.

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` makes the `cellab` command available (or use `python -m cellab`).

## Usage

```bash
cellab check automata/forest_fire.json experiments/forest_fire_brief.json   # validate files
cellab run experiments/forest_fire_brief.json                               # run + answer the questions
cellab run experiments/forest_fire_regrowth.json --seed 42 --trace          # print every step's fingerprint
cellab run experiments/rule30_ring16.json --stop-at 2000                    # stop early (or press Ctrl+C)
cellab resume runs/<experiment>/<date_time>                                 # continue where it stopped
cellab resume runs/<...> --steps 100                                        # run a finished run further
cellab answers runs/<...>                                                   # print the answers again
cellab show runs/<...> --step 3                                             # print one generation
cellab render runs/<...>                                                    # first.png, last.png, run.gif / spacetime.png
```

Each run gets a self-contained folder in `runs/` with copies of its input files (including the seed used), checkpoints, fingerprints, generations and answers.

## Datasets and neural networks (Deliverable 3)

```bash
cellab dataset datasets/elementary_w16.json --workers 4     # generate a dataset into data/
cellab qa data/elementary_w16.npz --limit 100               # the same as questions and answers in words
pip install -r requirements-ml.txt                          # PyTorch, only needed for the networks
python -m ml.train step data/elementary_w16.npz --out results/step --epochs 40
python -m ml.evaluate step data/elementary_w16.npz results/step/best.pt --out results/step
```

The datasets are described in [docs/DATASETS.md](docs/DATASETS.md), the experiment in [docs/deliverables/D3-dataset/README.md](docs/deliverables/D3-dataset/README.md). Training is meant to run on a GPU: open [notebooks/D3_train_on_colab.ipynb](notebooks/D3_train_on_colab.ipynb) in Google Colab.

## Documentation

- [docs/FILE_FORMATS.md](docs/FILE_FORMATS.md): how to write automaton and experiment files, including the rule language.
- [docs/GUIDE.md](docs/GUIDE.md): how the code is organised, how each part works, and where future changes go.
- Presentations: [Deliverable 1](docs/deliverables/D1-planning/CelLab-Deliverable1.pdf), [Deliverable 2](docs/deliverables/D2-modeling/CelLab-Deliverable2.pdf).

## Layout

| Folder | Contents |
| --- | --- |
| `src/cellab/` | The library and the `cellab` command |
| `automata/` | Automaton files: all 15 automata from the brief (10 update cells in place, 5 use the agents mode for things that move), plus Game of Life and any elementary rule |
| `experiments/` | Experiment files, using the brief's starting grids where it gives them |
| `datasets/` | Dataset files: which runs each machine-learning dataset consists of |
| `data/` | The generated datasets (`.npz` arrays and their `.json` descriptions) |
| `ml/` | The neural networks: models, training, evaluation (needs PyTorch) |
| `notebooks/` | The Colab notebook that trains the networks |
| `demo/` | Script and input files for the live demo of Deliverable 2 |
| `tests/` | 258 tests (`python -m pytest`) |
| `docs/` | File formats, code guide, deliverables, requirements |
| `docs/ai-logs/public/` | AI conversation logs, with personal details removed |

## AI conversation logging

The project requires every interaction with conversational agents to be saved.

- **Claude Code sessions opened in this folder** are saved automatically: a Stop hook (`.claude/settings.json`) runs `python tools/save_ai_log.py` after every reply. It writes the raw transcript to `docs/ai-logs/raw/` and a readable copy to `docs/ai-logs/readable/`; both stay local (git-ignored).
- **Other assistants:** export the conversation into `docs/ai-logs/raw/` as `YYYY-MM-DD_<assistant>_<topic>.md`.
- **Publishing:** `python tools/publish_ai_logs.py <session id>` writes a copy to `docs/ai-logs/public/` with names, e-mail addresses and account names replaced by placeholders. The texts to replace are listed in `docs/ai-logs/redact.txt`, which stays local; the script refuses to run without it. Run it again without arguments before each push to bring the published logs up to date, and read the result before committing.
