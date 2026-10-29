"""Plain helpers shared by the tests (fixtures live in conftest.py)."""

import json
from pathlib import Path

import numpy as np

from cellab import load_automaton, load_experiment, step
from cellab.randomness import make_generator

ROOT = Path(__file__).resolve().parent.parent
AUTOMATA = ROOT / "automata"
EXPERIMENTS = ROOT / "experiments"

FOREST_FIRE = {
    "name": "Test fire",
    "dimensions": 2,
    "states": ["EMPTY", "TREE", "FIRE"],
    "neighbourhood": "von_neumann",
    "edges": {"type": "fixed", "value": "EMPTY"},
    "rules": [
        {"when": "self == FIRE", "then": "EMPTY"},
        {"when": "self == TREE and count(FIRE) >= 1", "then": "FIRE"},
    ],
}


def write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def run_steps(automaton, grid, steps, seed=0):
    """Apply the engine `steps` times; returns the list of grids including the first."""
    rng = make_generator(seed)
    grids = [np.asarray(grid, dtype=automaton.states.dtype)]
    for k in range(steps):
        grids.append(step(automaton, grids[-1], rng, k))
    return grids


def example_automaton(name: str):
    return load_automaton(AUTOMATA / f"{name}.json")


def example_experiment(name: str):
    return load_experiment(EXPERIMENTS / f"{name}.json")


ELEMENTARY = json.loads((AUTOMATA / "elementary.json").read_text(encoding="utf-8"))


def run_experiment(experiment, tmp_path, **start_options):
    """Start and finish a run in a temporary runs folder. Returns (run, result)."""
    from cellab import Run
    run = Run.start(experiment, runs_dir=tmp_path / "runs", **start_options)
    return run, run.advance()


def answer_results(result) -> dict:
    """{question kind: result data} for a finished run (kinds are unique in our test experiments)."""
    return {answer.question["ask"]: answer.result for answer in result.answers}
