"""Nothing in the code is specific to one automaton.

The professor's requirement: the engine is general and everything about an
automaton comes from its input file. These tests check it mechanically.
"""

import json
import re

import pytest

from cellab import Run, load_automaton, load_experiment
from helpers import AUTOMATA, EXPERIMENTS, ROOT


def state_names() -> set[str]:
    names = set()
    for path in AUTOMATA.glob("*.json"):
        states = json.loads(path.read_text(encoding="utf-8"))["states"]
        if isinstance(states, list):
            names.update(name for name in states if len(name) >= 3)
    return names


def test_no_state_name_from_any_example_appears_in_the_code():
    names = state_names()
    assert len(names) > 10                          # the check is not vacuous
    for path in (ROOT / "src" / "cellab").glob("*.py"):
        code = path.read_text(encoding="utf-8")
        found = sorted(name for name in names if re.search(rf"\b{re.escape(name)}\b", code))
        assert not found, f"{path.name} mentions {found}"


@pytest.mark.parametrize("path", sorted(AUTOMATA.glob("*.json")), ids=lambda p: p.stem)
def test_every_example_automaton_loads(path):
    load_automaton(path)


@pytest.mark.parametrize("path", sorted(EXPERIMENTS.glob("*.json")), ids=lambda p: p.stem)
def test_every_example_experiment_runs_and_answers(path, tmp_path):
    experiment = load_experiment(path)
    run = Run.start(experiment, runs_dir=tmp_path)
    result = run.advance()
    assert len(result.answers) == len(experiment.questions)
    assert run.state.step >= experiment.steps
