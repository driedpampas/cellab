"""The rule language and the step semantics, checked on small hand-made grids."""

import numpy as np
import pytest

from cellab import step
from cellab.errors import RuleError
from cellab.randomness import make_generator

GRID = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])


def numbers(then, grid=GRID, edges=None, neighbourhood="von_neumann", parameters=None, when=None, step_number=0):
    """Build an integer automaton with a single rule, apply one step to `grid`."""
    from cellab.automaton import load_automaton
    from helpers import write_json
    import tempfile
    from pathlib import Path
    rule = {"then": then} if when is None else {"when": when, "then": then}
    data = {"name": "probe", "dimensions": np.ndim(grid), "states": {"type": "integer"},
            "neighbourhood": neighbourhood, "edges": edges or {"type": "fixed", "value": 0},
            "parameters": parameters or {}, "rules": [rule]}
    with tempfile.TemporaryDirectory() as folder:
        automaton = load_automaton(write_json(Path(folder) / "probe.json", data))
    return step(automaton, np.asarray(grid), make_generator(0), step_number).tolist()


# ------------------------------------------------------------ names and neighbours

def test_self_and_arithmetic():
    assert numbers("self * 2 + 1") == (GRID * 2 + 1).tolist()
    assert numbers("self // 4") == (GRID // 4).tolist()
    assert numbers("self % 3") == (GRID % 3).tolist()
    assert numbers("self ** 2 - self") == (GRID ** 2 - GRID).tolist()


def test_single_neighbours_with_fixed_edges():
    assert numbers("north") == [[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]]
    assert numbers("south") == [[5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0]]
    assert numbers("west") == [[0, 1, 2, 3], [0, 5, 6, 7], [0, 9, 10, 11]]
    assert numbers("east") == [[2, 3, 4, 0], [6, 7, 8, 0], [10, 11, 12, 0]]
    assert numbers("north_west")[1] == [0, 1, 2, 3]


def test_wrap_edges_connect_opposite_sides():
    wrap = {"type": "wrap"}
    assert numbers("north", edges=wrap)[0] == [9, 10, 11, 12]       # top row sees the bottom row
    assert numbers("west", edges=wrap)[1][0] == 8                   # left column sees the right column
    assert numbers("south_east", edges=wrap)[2][3] == 1             # the corner sees the opposite corner


def test_fixed_edge_value_is_used_beyond_the_grid():
    assert numbers("north", edges={"type": "fixed", "value": 7})[0] == [7, 7, 7, 7]


def test_left_and_right_in_one_dimension():
    row = [1, 2, 3, 4]
    assert numbers("left", grid=row, neighbourhood="elementary") == [0, 1, 2, 3]
    assert numbers("right", grid=row, neighbourhood="elementary", edges={"type": "wrap"}) == [2, 3, 4, 1]


def test_aggregates_over_the_neighbourhood():
    assert numbers("sum(nb)")[1][1] == 2 + 5 + 7 + 10
    assert numbers("sum(nb)", neighbourhood="moore")[1][1] == 1 + 2 + 3 + 5 + 7 + 9 + 10 + 11
    assert numbers("min(nb)")[1][1] == 2
    assert numbers("max(nb)")[1][1] == 10
    assert numbers("count(nb > 6)")[1][1] == 2           # 7 and 10
    assert numbers("round(mean(nb))")[1][1] == 6           # (2+5+7+10)/4 = 6
    assert numbers("count(nb == self + 1)")[1][1] == 1     # only 7 equals 6 + 1


def test_custom_neighbourhood_offsets():
    assert numbers("sum(nb)", neighbourhood=[[0, 2], [2, 0]])[0][0] == 3 + 9


def test_elementwise_min_max_and_functions():
    assert numbers("min(self, 5)") == np.minimum(GRID, 5).tolist()
    assert numbers("max(self, north, 3)")[0][0] == 3
    assert numbers("abs(self - 6)")[0] == [5, 4, 3, 2]
    assert numbers("floor(self / 4)") == (GRID // 4).tolist()
    assert numbers("round(self / 4)")[0] == [0, 0, 1, 1]       # 0.25, 0.5 (to even), 0.75, 1


def test_conditions_and_choices():
    assert numbers("1 if self > 6 else 0") == (GRID > 6).astype(int).tolist()
    assert numbers("2 < self <= 5")[0] == [0, 0, 1, 1]         # chained comparison
    assert numbers("self > 2 and not self > 10")[2] == [1, 1, 0, 0]
    assert numbers("self < 2 or self > 11")[0][0] == 1
    assert numbers("self & 1")[0] == [1, 0, 1, 0]                # bit operators
    assert numbers("(30 >> self) & 1", grid=[0, 1, 2, 3, 4], neighbourhood="elementary") == [0, 1, 1, 1, 1]


def test_parameters_and_step_number():
    assert numbers("self + k", parameters={"k": 100})[0][0] == 101
    assert numbers("step", step_number=41)[0][0] == 41


def test_when_limits_which_cells_change():
    assert numbers("0", when="self % 2 == 0")[0] == [1, 0, 3, 0]


# ------------------------------------------------------------ named states and rule order

def test_first_matching_rule_wins_and_otherwise_applies(make_automaton):
    automaton = make_automaton(states=["A", "B", "C"], edges={"type": "wrap"}, rules=[
        {"when": "self == A", "then": "B"},
        {"when": "self == A or self == B", "then": "C"},   # A cells were already decided by rule 0
    ], otherwise="A")
    grid = np.array([[0, 1, 2]], dtype=np.uint8)
    assert step(automaton, grid, make_generator(0), 0).tolist() == [[1, 2, 0]]


def test_count_shorthand_equals_explicit_count(make_automaton):
    short = make_automaton(rules=[{"when": "count(FIRE) >= 2", "then": "FIRE"}])
    explicit = make_automaton(rules=[{"when": "count(nb == FIRE) >= 2", "then": "FIRE"}])
    grid = np.random.default_rng(3).integers(0, 3, size=(8, 8)).astype(np.uint8)
    rng = make_generator(0)
    assert (step(short, grid, rng, 0) == step(explicit, grid, rng, 0)).all()


# ------------------------------------------------------------ errors while running

def test_value_outside_the_named_states_is_reported(make_automaton):
    automaton = make_automaton(rules=[{"when": "self == FIRE", "then": "self + 1"}])
    with pytest.raises(RuleError, match="above the highest allowed value FIRE"):
        step(automaton, np.array([[2]], dtype=np.uint8), make_generator(0), 0)


def test_no_silent_wraparound_below_zero(make_automaton):
    automaton = make_automaton(rules=[{"then": "self - 1"}])
    with pytest.raises(RuleError, match="value -1, below the lowest allowed value EMPTY"):
        step(automaton, np.array([[0]], dtype=np.uint8), make_generator(0), 0)


def test_whole_number_states_reject_fractions():
    with pytest.raises(RuleError, match="not a whole number"):
        numbers("self / 2")


def test_fractions_are_fine_where_the_rule_does_not_apply():
    assert numbers("self / 2", when="self % 2 == 0")[0] == [1, 1, 3, 2]


def test_division_by_zero_is_reported_with_the_rule():
    with pytest.raises(RuleError, match=r"rules\[0\]\.then .*divide by zero"):
        numbers("1 // (self - self)")


def test_integer_limits_are_enforced(make_automaton):
    automaton = make_automaton(states={"type": "integer", "min": 0, "max": 3},
                               edges={"type": "wrap"}, rules=[{"then": "self + 1"}])
    with pytest.raises(RuleError, match="above the highest allowed value 3"):
        step(automaton, np.array([[3]]), make_generator(0), 0)
