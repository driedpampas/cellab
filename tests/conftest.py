"""Pytest fixtures shared by the tests."""

import pytest

from cellab import load_automaton, load_experiment
from helpers import FOREST_FIRE, write_json


@pytest.fixture
def make_automaton(tmp_path):
    """Write an automaton dict to a file and load it: make_automaton(rules=[...]) etc."""
    counter = iter(range(10_000))

    def make(base=None, **changes):
        data = dict(base or FOREST_FIRE, **changes)
        return load_automaton(write_json(tmp_path / f"automaton_{next(counter)}.json", data))
    return make


@pytest.fixture
def make_experiment(tmp_path):
    """Write an experiment (and optionally its automaton) and load it."""
    counter = iter(range(10_000))

    def make(automaton=None, **fields):
        number = next(counter)
        automaton_file = write_json(tmp_path / f"exp_automaton_{number}.json", automaton or FOREST_FIRE)
        data = {"name": f"test {number}", "automaton": automaton_file.name, "steps": 5, **fields}
        return load_experiment(write_json(tmp_path / f"experiment_{number}.json", data))
    return make
