"""Results known in advance: from the project brief, the lab notes, the Deliverable 1
slides, or mathematics. Each one is checked through a real run."""

import numpy as np
import pytest

from helpers import ELEMENTARY, answer_results, example_automaton, example_experiment, run_experiment, run_steps


def test_forest_fire_from_the_brief(tmp_path):
    experiment = example_experiment("forest_fire_brief")
    run, result = run_experiment(experiment, tmp_path)
    assert experiment.format(run.grid_at(1)) == ["TTTT.", "TT.TT", ".F.FT", "TTFTT"]
    assert experiment.format(run.grid_at(2)) == ["TTTT.", "TF.FT", "....F", "TF.FT"]
    assert experiment.format(run.grid_at(3)) == ["TFTF.", "F...F", ".....", "F...F"]
    results = answer_results(result)
    assert results["stable"] == {"stable": True, "from_step": 5}
    assert results["count_after"]["count"] == 2
    assert results["occurs"] == {"occurs": False, "certain": True}
    assert results["cycle"] == {"cycle": True, "start": 5, "length": 1}


def test_sandpile_from_the_brief(tmp_path):
    experiment = example_experiment("sandpile_brief")
    run, result = run_experiment(experiment, tmp_path)
    assert answer_results(result)["stable"]["from_step"] == 3
    assert run.grid_at(3).tolist() == [[1, 3, 2], [0, 2, 3], [3, 0, 1]]


def test_rule58_example_from_the_lab_notes(tmp_path):
    run, _ = run_experiment(example_experiment("rule58_lab"), tmp_path)
    assert "".join(map(str, run.grid_at(1))) == "0011010"


@pytest.mark.parametrize("width, start, length", [(8, 1, 40), (12, 24, 102), (14, 0, 1428), (16, 58, 6016)])
def test_rule30_ring_cycles_from_the_deliverable1_slides(make_experiment, tmp_path, width, start, length):
    experiment = make_experiment(
        automaton=dict(ELEMENTARY, parameters={"rule": 30}), size=[width], steps=0, max_steps=20000,
        initial={"type": "single", "value": 1, "background": 0}, store_generations=False,
        questions=[{"ask": "cycle"}])
    _, result = run_experiment(experiment, tmp_path)
    assert answer_results(result)["cycle"] == {"cycle": True, "start": start, "length": length}


@pytest.mark.parametrize("rule, row, count, cycle_start, cycle_length", [
    (30, "1010001000011000", 9, 30, 4144),
    (90, "1000010000110010", 0, 8, 1),
    (110, "0010000111111100", 8, 50, 32),
    (184, "0011111001010110", 9, 3, 16),
    (250, "0111110011001111", 16, 1, 1),
])
def test_dataset_rows_from_the_deliverable1_slides(make_experiment, tmp_path, rule, row, count,
                                                   cycle_start, cycle_length):
    experiment = make_experiment(
        automaton=dict(ELEMENTARY, parameters={"rule": rule}), steps=0, max_steps=10000,
        initial={"type": "text", "rows": [row]}, store_generations=False,
        questions=[{"ask": "count_after", "n": 50, "value": 1}, {"ask": "cycle"}])
    _, result = run_experiment(experiment, tmp_path)
    results = answer_results(result)
    assert results["count_after"]["count"] == count
    assert (results["cycle"]["start"], results["cycle"]["length"]) == (cycle_start, cycle_length)


def test_rule90_on_a_16_cell_ring_dies_within_8_steps(make_automaton):
    """Rule 90 is x -> left XOR right. On 2^k cells, ANY start is all zeros after 2^(k-1) steps."""
    automaton = make_automaton(ELEMENTARY, parameters={"rule": 90})
    rng = np.random.default_rng(90)
    for _ in range(50):
        rows = run_steps(automaton, rng.integers(0, 2, size=16), 8)
        assert not rows[8].any()


def test_traffic_keeps_its_cars():
    rows = run_steps(example_automaton("traffic"), np.random.default_rng(1).integers(0, 2, size=50), 100)
    assert len({int(row.sum()) for row in rows}) == 1


def test_glider_returns_after_48_steps_on_a_12x12_torus(tmp_path):
    _, result = run_experiment(example_experiment("game_of_life_glider"), tmp_path)
    assert answer_results(result)["cycle"] == {"cycle": True, "start": 0, "length": 48}


def test_diffusion_keeps_the_total_heat_up_to_rounding(tmp_path):
    run, _ = run_experiment(example_experiment("diffusion_brief"), tmp_path)
    for k in range(13):
        assert abs(run.grid_at(k).sum() - 100) < 0.1
