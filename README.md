# CelLab

AI course project: a system for describing cellular automata in natural language,
simulating them, and answering questions about their evolution.

Team: Alexandru Bianca Ana-Maria, Gurzu Matei.

## Layout

| Folder | Contents |
| --- | --- |
| `src/` | Application source code |
| `data/` | Generated datasets (generations, Q/A sets for training) |
| `docs/requirements/` | Project brief and lab notes |
| `docs/deliverables/` | Material for each lab deliverable (D1 to D4) |
| `docs/ai-logs/` | AI conversation logs, **local only** (git-ignored, see below) |
| `tools/` | Helper scripts (e.g. `save_ai_log.py`) |

Deliverable 1 presentation: [`docs/deliverables/D1-planning/CelLab-Deliverable1.pdf`](docs/deliverables/D1-planning/CelLab-Deliverable1.pdf)

## AI conversation logging

The project requires every interaction with conversational agents to be saved.
The logs are **not pushed** to this public repository; each of us keeps them locally
in `docs/ai-logs/` and they are handed in privately at submission.

- **Claude Code sessions opened in this folder** are saved automatically: a Stop hook
  (`.claude/settings.json`) runs `python tools/save_ai_log.py` after every reply and
  writes a raw transcript (`docs/ai-logs/raw/`) and a readable copy (`docs/ai-logs/readable/`).
- **Other assistants** (ChatGPT, Gemini, Colab's assistant, etc.): export or copy the
  conversation into `docs/ai-logs/raw/` as `YYYY-MM-DD_<assistant>_<topic>.md`.
