"""The engine against independent, slow reference implementations, on random grids.

reference.py implements every example automaton with plain loops and no shared
code. Agreement over many random grids and steps is strong evidence that both
the rule files and the engine do what the brief describes.
"""

import numpy as np
import pytest

import reference
from helpers import ELEMENTARY, example_automaton, run_steps

CASES_2D = [
    ("forest_fire", reference.forest_fire, [0.3, 0.6, 0.1]),
    ("brians_brain", reference.brians_brain, [0.6, 0.3, 0.1]),
    ("game_of_life", reference.game_of_life, [0.65, 0.35]),
    ("sandpile", reference.sandpile, None),
    ("cyclic", reference.cyclic, [1, 1, 1, 1]),
    ("voter", reference.voter, [1, 1]),
    ("crystal_growth", reference.crystal_growth, [0.9, 0.1]),
    ("epidemic", reference.epidemic, [0.8, 0.1, 0.05, 0.03, 0.02]),
]


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("name, reference_step, weights", CASES_2D)
def test_2d_automata_match_reference(name, reference_step, weights, seed):
    automaton = example_automaton(name)
    test_rng = np.random.default_rng(seed)
    shape = (9, 13)                      # not square, to catch mixed-up rows and columns
    if weights is None:
        grid = test_rng.integers(0, 8, size=shape)
    else:
        grid = test_rng.choice(len(weights), size=shape, p=np.array(weights) / sum(weights))
    engine_grids = run_steps(automaton, grid, 15)
    expected = grid.tolist()
    for k in range(1, 16):
        expected = reference_step(expected)
        assert engine_grids[k].tolist() == expected, f"{name}: step {k} differs"


def test_diffusion_matches_reference():
    automaton = example_automaton("diffusion")
    grid = np.random.default_rng(4).integers(0, 100, size=(6, 7)).astype(float)
    engine_grids = run_steps(automaton, grid, 12)
    expected = grid.tolist()
    for k in range(1, 13):
        expected = reference.diffusion(expected)
        assert np.allclose(engine_grids[k], expected, atol=1e-9), f"step {k} differs"


@pytest.mark.parametrize("rule", [30, 54, 90, 110, 150, 184, 250])
def test_elementary_rules_match_reference(make_automaton, rule):
    automaton = make_automaton(ELEMENTARY, parameters={"rule": rule})
    row = np.random.default_rng(rule).integers(0, 2, size=37)
    engine_rows = run_steps(automaton, row, 30)
    expected = row.tolist()
    for k in range(1, 31):
        expected = reference.elementary(expected, rule, wrap=True)
        assert engine_rows[k].tolist() == expected, f"rule {rule}: step {k} differs"


def test_rule30_table_with_fixed_edges_matches_reference():
    automaton = example_automaton("rule30")
    row = np.random.default_rng(30).integers(0, 2, size=41)
    engine_rows = run_steps(automaton, row, 30)
    expected = row.tolist()
    for k in range(1, 31):
        expected = reference.elementary(expected, 30, wrap=False, edge=0)
        assert engine_rows[k].tolist() == expected


def test_traffic_matches_reference():
    automaton = example_automaton("traffic")
    row = np.random.default_rng(184).integers(0, 2, size=40)
    engine_rows = run_steps(automaton, row, 30)
    expected = row.tolist()
    for k in range(1, 31):
        expected = reference.traffic(expected)
        assert engine_rows[k].tolist() == expected
