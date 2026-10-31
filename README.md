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

## Documentation

- [docs/FILE_FORMATS.md](docs/FILE_FORMATS.md): how to write automaton and experiment files, including the rule language.
- [docs/GUIDE.md](docs/GUIDE.md): how the code is organised, how each part works, and where future changes go.

## Layout

| Folder | Contents |
| --- | --- |
| `src/cellab/` | The library and the `cellab` command |
| `automata/` | Automaton files: 10 of the brief's 15 automata (all that update cells in place; the other 5 move agents around), plus Game of Life and any elementary rule |
| `experiments/` | Experiment files, using the brief's starting grids where it gives them |
| `demo/` | Script and input files for the live demo of Deliverable 2 |
| `tests/` | 195 tests (`python -m pytest`) |
| `docs/` | File formats, code guide, deliverables, requirements |
| `docs/ai-logs/` | AI conversation logs, **local only** (git-ignored) |

## AI conversation logging

The project requires every interaction with conversational agents to be saved. The logs are **not pushed** to this public repository; each of us keeps them locally in `docs/ai-logs/`, and they are handed in privately at submission.

- **Claude Code sessions opened in this folder** are saved automatically: a Stop hook (`.claude/settings.json`) runs `python tools/save_ai_log.py` after every reply.
- **Other assistants:** export the conversation into `docs/ai-logs/raw/` as `YYYY-MM-DD_<assistant>_<topic>.md`.
