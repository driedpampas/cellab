"""Fixed randomness and resuming: the two "maximum score" guarantees.

1. The same seed always gives the same run; different seeds give different runs.
2. A run that is stopped (Ctrl+C, stop_at, or even a crash) and then resumed gives
   exactly the same fingerprints, generations and answers as a run never stopped.
"""

import re
import signal
from pathlib import Path

import numpy as np
import pytest

import cellab.runner
from cellab import Run, StorageError
from cellab.storage import RunFolder
from helpers import ROOT, example_experiment

RANDOM_FOREST = {
    "name": "random forest",
    "dimensions": 2,
    "states": ["EMPTY", "TREE", "FIRE"],
    "neighbourhood": "von_neumann",
    "edges": {"type": "wrap"},
    "parameters": {"p": 0.05, "f": 0.002},
    "rules": [
        {"when": "self == FIRE", "then": "EMPTY"},
        {"when": "self == TREE and (count(FIRE) >= 1 or random() < f)", "then": "FIRE"},
        {"when": "self == EMPTY and random() < p", "then": "TREE"},
    ],
}


@pytest.fixture
def random_experiment(make_experiment):
    def make(**fields):
        settings = dict(size=[20, 30], steps=60, max_steps=75, checkpoint_every=10,
                        initial={"type": "random", "weights": {"TREE": 0.6, "EMPTY": 0.4}},
                        questions=[{"ask": "count_after", "n": 60, "value": "TREE"}, {"ask": "cycle"}])
        settings.update(fields)
        return make_experiment(automaton=RANDOM_FOREST, **settings)
    return make


def folder_files(run: Run) -> dict[str, bytes]:
    names = [RunFolder.FINGERPRINTS, RunFolder.GENERATIONS, RunFolder.ANSWERS]
    return {name: run.folder.file(name).read_bytes() for name in names if run.folder.file(name).exists()}


# ------------------------------------------------------------ same seed, same run

def test_same_seed_gives_identical_runs(random_experiment, tmp_path):
    experiment = random_experiment()
    first = Run.start(experiment, runs_dir=tmp_path / "a", seed=123)
    first.advance()
    second = Run.start(experiment, runs_dir=tmp_path / "b", seed=123)
    second.advance()
    assert first.state.history.fingerprints == second.state.history.fingerprints
    assert folder_files(first) == folder_files(second)


def test_different_seeds_give_different_runs(random_experiment, tmp_path):
    experiment = random_experiment()
    first = Run.start(experiment, runs_dir=tmp_path / "a", seed=1)
    first.advance()
    second = Run.start(experiment, runs_dir=tmp_path / "b", seed=2)
    second.advance()
    assert first.state.history.fingerprints != second.state.history.fingerprints


def test_the_seed_is_recorded_even_when_the_file_has_none(random_experiment, tmp_path):
    run = Run.start(random_experiment(), runs_dir=tmp_path)
    run.advance()
    import json
    recorded = json.loads(run.folder.file(RunFolder.EXPERIMENT).read_text())["seed"]
    assert recorded == run.state.seed
    again = Run.start(random_experiment(), runs_dir=tmp_path, seed=recorded)
    again.advance()
    assert again.state.history.fingerprints == run.state.history.fingerprints


def test_no_other_source_of_randomness_in_the_code():
    """Only randomness.py may create generators; nothing may use global random functions."""
    for path in (ROOT / "src" / "cellab").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import random|from random import)", text, re.M), path.name
        if path.name != "randomness.py":
            assert not re.search(r"np\.random\.(?!Generator\b)\w+", text), path.name


# ------------------------------------------------------------ stop and resume

def stop_and_resume(experiment, tmp_path, stops):
    """Run with stops at the given steps, reopening the folder each time."""
    run = Run.start(experiment, runs_dir=tmp_path, seed=7)
    for stop in stops:
        result = run.advance(stop_at=stop)
        assert result.stopped and run.state.step == stop
        run = Run.open(run.folder.path)
        assert run.state.step == stop
    result = run.advance()
    assert not result.stopped
    return run


@pytest.mark.parametrize("stops", [[1], [23], [10, 20, 33], [59]])
def test_resumed_random_run_equals_uninterrupted_run(random_experiment, tmp_path, stops):
    experiment = random_experiment()
    straight = Run.start(experiment, runs_dir=tmp_path / "straight", seed=7)
    straight.advance()
    resumed = stop_and_resume(experiment, tmp_path / "resumed", stops)
    assert resumed.state.history.fingerprints == straight.state.history.fingerprints
    assert folder_files(resumed) == folder_files(straight)


def test_resumed_deterministic_run_equals_uninterrupted_run(tmp_path):
    experiment = example_experiment("rule30_ring16")          # 6074 steps, generations not stored
    straight = Run.start(experiment, runs_dir=tmp_path / "straight", seed=1)
    straight.advance()
    resumed = stop_and_resume(experiment, tmp_path / "resumed", [1000, 4321])
    assert resumed.state.history.fingerprints == straight.state.history.fingerprints
    assert folder_files(resumed) == folder_files(straight)


def test_resume_after_a_crash_between_checkpoints(random_experiment, tmp_path, monkeypatch):
    """A crash loses only the steps after the last checkpoint; they are recomputed identically."""
    experiment = random_experiment()
    straight = Run.start(experiment, runs_dir=tmp_path / "straight", seed=7)
    straight.advance()

    real_step, calls = cellab.runner.step, {"n": 0}

    def crashing_step(*args):
        calls["n"] += 1
        if calls["n"] == 25:
            raise RuntimeError("power cut")
        return real_step(*args)

    crashed = Run.start(experiment, runs_dir=tmp_path / "crashed", seed=7)
    monkeypatch.setattr(cellab.runner, "step", crashing_step)
    with pytest.raises(RuntimeError, match="power cut"):
        crashed.advance()
    monkeypatch.setattr(cellab.runner, "step", real_step)

    reopened = Run.open(crashed.folder.path)
    assert reopened.state.step == 20                      # the last checkpoint (checkpoint_every = 10)
    reopened.advance()
    assert folder_files(reopened) == folder_files(straight)


def test_ctrl_c_stops_cleanly_between_steps(random_experiment, tmp_path):
    experiment = random_experiment()
    straight = Run.start(experiment, runs_dir=tmp_path / "straight", seed=7)
    straight.advance()

    run = Run.start(experiment, runs_dir=tmp_path / "interrupted", seed=7)

    def press_ctrl_c_at_step_17(state):
        if state.step == 17:
            signal.raise_signal(signal.SIGINT)

    result = run.advance(on_step=press_ctrl_c_at_step_17)
    assert result.stopped and run.state.step == 17
    assert run.folder.read_status()["status"] == "stopped"
    Run.open(run.folder.path).advance()
    assert folder_files(Run.open(run.folder.path)) == folder_files(straight)


def test_resume_refuses_an_edited_automaton(random_experiment, tmp_path):
    run = Run.start(random_experiment(), runs_dir=tmp_path, seed=7)
    run.advance(stop_at=5)
    automaton_copy = run.folder.file(RunFolder.AUTOMATON)
    automaton_copy.write_text(automaton_copy.read_text().replace("0.05", "0.5"))
    with pytest.raises(StorageError, match="was changed after the run started"):
        Run.open(run.folder.path)


def test_a_finished_run_can_be_extended(tmp_path):
    experiment = example_experiment("rule58_lab")                # finishes at step 10
    longer = Run.start(experiment.with_steps(25), runs_dir=tmp_path / "longer")
    longer.advance()
    run = Run.start(experiment, runs_dir=tmp_path / "extended")
    run.advance()
    reopened = Run.open(run.folder.path)
    reopened.extend(15)
    reopened.advance()
    assert reopened.state.step == 25
    assert reopened.state.history.fingerprints == longer.state.history.fingerprints


def test_opening_something_that_is_not_a_run_folder(tmp_path):
    with pytest.raises(StorageError, match="not a run folder"):
        Run.open(tmp_path)
