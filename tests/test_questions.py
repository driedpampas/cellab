"""The five questions, including the cases that need care: jumping ahead through a
cycle, a certain "never", random rules, and honest "not decided" answers."""

import numpy as np

import reference
from helpers import ELEMENTARY, answer_results, example_experiment, run_experiment

SINGLE_CELL = {"type": "single", "value": 1, "background": 0}


def test_state_after_far_beyond_the_run_uses_the_cycle(make_experiment, tmp_path):
    """Rule 30 on 8 cells enters a cycle of 40 at step 1; step 12345 is computed without simulating it."""
    n = 12_345
    experiment = make_experiment(automaton=dict(ELEMENTARY, parameters={"rule": 30}), size=[8], steps=0,
                                 initial=SINGLE_CELL, questions=[{"ask": "state_after", "n": n}])
    run, result = run_experiment(experiment, tmp_path)
    assert run.state.step < 50                                   # the run itself stayed short
    row = [0, 0, 0, 0, 1, 0, 0, 0]
    for _ in range(n):                                           # slow, direct simulation for comparison
        row = reference.elementary(row, 30)
    answer = answer_results(result)["state_after"]
    assert answer["grid"] == ["".join(map(str, row))]
    assert answer["same_as_step"] == 1 + (n - 1) % 40


def test_state_after_zero_is_the_initial_world(tmp_path):
    experiment = example_experiment("forest_fire_brief")
    run, _ = run_experiment(experiment, tmp_path)
    assert experiment.format(run.grid_at(0)) == ["TTTT.", "TT.TT", ".TFTT", "TTTTT"]


def test_occurs_finds_the_first_step(make_experiment, tmp_path):
    experiment = make_experiment(
        initial={"type": "text", "rows": ["TTTT.", "TT.TT", ".TFTT", "TTTTT"],
                 "legend": {"T": "TREE", "F": "FIRE", ".": "EMPTY"}},
        questions=[{"ask": "occurs", "rows": ["TTTT.", "TF.FT", "....F", "TF.FT"]}])
    _, result = run_experiment(experiment, tmp_path)
    assert answer_results(result)["occurs"] == {"occurs": True, "first_step": 2}


def test_questions_that_cannot_be_decided_say_so(make_experiment, tmp_path):
    experiment = make_experiment(automaton=dict(ELEMENTARY, parameters={"rule": 30}), size=[40], steps=0,
                                 max_steps=50, initial=SINGLE_CELL,
                                 questions=[{"ask": "cycle"}, {"ask": "stable"}, {"ask": "state_after", "n": 999}])
    run, result = run_experiment(experiment, tmp_path)
    assert run.state.step == 50
    assert all(not answer.decided for answer in result.answers)
    assert all(answer.text.startswith("Not decided") for answer in result.answers)


def test_random_rules_never_claim_a_guaranteed_cycle(make_experiment, tmp_path):
    automaton = {
        "name": "coin", "dimensions": 1, "states": {"type": "integer", "min": 0, "max": 1},
        "neighbourhood": "elementary", "edges": {"type": "wrap"},
        "rules": [{"then": "1 if random() < 0.5 else 0"}],
    }
    experiment = make_experiment(automaton=automaton, size=[2], steps=0, max_steps=200, seed=5,
                                 initial={"type": "text", "rows": ["00"]},
                                 questions=[{"ask": "cycle"}, {"ask": "stable"}])
    _, result = run_experiment(experiment, tmp_path)
    for answer in result.answers:
        assert "random numbers" in answer.text           # 2 cells: a repeat is found quickly
    assert answer_results(result)["cycle"]["guaranteed"] is False


def test_count_after_uses_the_requested_value(tmp_path):
    _, result = run_experiment(example_experiment("traffic_brief"), tmp_path)
    assert answer_results(result)["count_after"] == {"step": 1000, "value": "CAR", "count": 4, "same_as_step": 10}


def test_answers_are_saved_in_the_run_folder(tmp_path):
    run, result = run_experiment(example_experiment("forest_fire_brief"), tmp_path)
    saved = run.folder.read_answers()
    assert saved == [answer.to_dict() for answer in result.answers]
    assert run.folder.read_status()["status"] == "finished"
