"""The agents mode: automata where things move, eat, multiply or die.

The first part checks the mechanics on tiny hand-made automata, the second the five
automata from the brief that need this mode, the third that layers are stored,
resumed and fingerprinted like everything else.
"""

import numpy as np
import pytest

import reference
from cellab import InputFileError, Run, RuleError, make_initial_grid, step
from cellab.fingerprints import FINGERPRINT_SIZE
from cellab.randomness import make_generator
from cellab.storage import RunFolder
from helpers import example_automaton, example_experiment, run_experiment, run_steps

# A row of cells in which things ("X") move to the right.
MOVERS = {
    "name": "movers",
    "dimensions": 1,
    "states": ["FREE", "X", "WALL"],
    "neighbourhood": "elementary",
    "edges": {"type": "fixed", "value": "WALL"},
    "update": "agents",
    "order": "reading",
    "rules": [{"when": "self == X",
               "do": {"to": {"among": ["right"], "where": "nb == FREE", "choose": "first"},
                      "here": {"self": "FREE"}, "there": {"self": "X"}}}],
}


def row(automaton, text: str) -> np.ndarray:
    return np.array([{".": 0, "X": 1, "#": 2}[ch] for ch in text], dtype=automaton.states.dtype)


def text(grid) -> str:
    return "".join(".X#"[v] for v in grid)


# ------------------------------------------------------------ mechanics

def test_a_thing_moves_one_cell_per_step_and_stops_at_the_edge(make_automaton):
    automaton = make_automaton(MOVERS)
    grids = run_steps(automaton, row(automaton, "X..."), 5)
    assert [text(g) for g in grids] == ["X...", ".X..", "..X.", "...X", "...X", "...X"]


def test_a_cell_something_moved_into_waits_until_the_next_step(make_automaton):
    # In reading order the first X moves into the second cell; if that cell acted again
    # in the same step, the X would run to the end of the row at once.
    automaton = make_automaton(MOVERS)
    assert text(run_steps(automaton, row(automaton, "X....."), 1)[1]) == ".X...."


def test_the_order_decides_who_goes_first(make_automaton):
    # Two things in a row: the one in front must move first for the one behind to follow.
    reading = make_automaton(MOVERS)
    reverse = make_automaton(MOVERS, order="reverse")
    assert text(run_steps(reading, row(reading, "XX.."), 1)[1]) == "X.X."     # the first X was still blocked
    assert text(run_steps(reverse, row(reverse, "XX.."), 1)[1]) == ".XX."    # the front X moved, then the other


def test_who_acts_is_decided_from_the_old_grid(make_automaton):
    # Which rule a cell follows is fixed from the grid at the START of the step; what the
    # action finds is the live grid. Here the middle cell fills up first (reverse order),
    # so the X on the left, which had decided to move, finds its way blocked.
    rules = MOVERS["rules"] + [{"when": "self == FREE and left == X and right == X", "then": "X"}]
    automaton = make_automaton(MOVERS, rules=rules, order="reverse")
    assert text(run_steps(automaton, row(automaton, "X.X#"), 1)[1]) == "XXX#"
    # In reading order the left X moves in first, and the middle cell loses its turn.
    automaton = make_automaton(MOVERS, rules=rules)
    assert text(run_steps(automaton, row(automaton, "X.X#"), 1)[1]) == ".XX#"


def test_actions_are_tried_in_order_until_one_is_possible(make_automaton):
    rules = [{"when": "self == X", "do": [
        {"to": {"among": ["right"], "where": "nb == FREE"}, "here": {"self": "FREE"}, "there": {"self": "X"}},
        {"to": {"among": ["left"], "where": "nb == FREE"}, "here": {"self": "FREE"}, "there": {"self": "X"}},
        {"here": {"self": "WALL"}}]}]
    automaton = make_automaton(MOVERS, rules=rules)
    assert text(run_steps(automaton, row(automaton, ".X."), 1)[1]) == "..X"      # right is free
    assert text(run_steps(automaton, row(automaton, ".X#"), 1)[1]) == "X.#"      # right blocked: left
    assert text(run_steps(automaton, row(automaton, "#X#"), 1)[1]) == "###"      # both blocked: last action


def test_random_choice_follows_the_seed(make_automaton):
    rules = [{"when": "self == X", "do": {"to": {"where": "nb == FREE"}, "here": {"self": "FREE"}, "there": {"self": "X"}}}]
    automaton = make_automaton(MOVERS, rules=rules, edges={"type": "wrap"})
    assert not automaton.deterministic
    start = row(automaton, "....X....")
    first, again = run_steps(automaton, start, 20, seed=3), run_steps(automaton, start, 20, seed=3)
    assert all((a == b).all() for a, b in zip(first, again))
    assert {text(run_steps(automaton, start, 1, seed=s)[1]) for s in range(20)} == {"...X.....", ".....X..."}


def test_anywhere_reaches_the_whole_grid(make_automaton):
    rules = [{"when": "self == X", "do": {"to": {"among": "anywhere", "where": "nb == FREE", "choose": "first"},
                                          "here": {"self": "WALL"}, "there": {"self": "X"}}}]
    automaton = make_automaton(MOVERS, rules=rules)
    assert text(run_steps(automaton, row(automaton, "##X#.#"), 1)[1]) == "####X#"


def test_then_is_short_for_changing_the_cell_itself(make_automaton):
    automaton = make_automaton(MOVERS, rules=[{"when": "self == X and right == WALL", "then": "WALL"}])
    assert text(run_steps(automaton, row(automaton, "X.X#"), 2)[2]) == "X.##"


LAYERED = dict(MOVERS, layers={"fuel": {"max": 9}, "trips": {}}, rules=[
    {"when": "self == X",
     "do": [{"to": {"among": ["right"], "where": "nb == FREE and fuel > 0", "choose": "first"},
             "here": {"self": "FREE", "fuel": 0, "trips": 0},
             "there": {"self": "X", "fuel": "fuel - 1", "trips": "trips + direction + 1"}},
            {"here": {"self": "WALL"}}]}])


def test_layers_travel_with_what_moves(make_automaton, make_experiment):
    experiment = make_experiment(LAYERED, initial={"type": "text", "rows": ["X....."], "legend": {".": "FREE"},
                                                   "layers": {"fuel": "3 * (self == X)"}})
    automaton = experiment.automaton
    states = [make_initial_grid(experiment, make_generator(0))]
    for k in range(4):
        states.append(step(automaton, states[-1], make_generator(0), k))
    assert states[0].shape == (3, 6) and states[0].dtype == np.int64        # "trips" has no maximum
    assert [text(automaton.cells(s)) for s in states] == ["X.....", ".X....", "..X...", "...X..", "...#.."]
    assert [int(automaton.layer(s, "fuel").sum()) for s in states] == [3, 2, 1, 0, 0]
    assert automaton.layer(states[3], "trips").tolist() == [0, 0, 0, 3, 0, 0]


def test_layer_values_outside_their_range_are_reported(make_automaton, make_experiment):
    rules = [{"when": "self == X", "do": {"here": {"fuel": "fuel + 5"}}}]
    experiment = make_experiment(dict(LAYERED, rules=rules), initial={
        "type": "text", "rows": ["X."], "legend": {".": "FREE"}, "layers": {"fuel": {"type": "text", "rows": ["7 0"]}}})
    with pytest.raises(RuleError, match=r"step 0 -> 1: rules\[0\].do.here.fuel \('fuel \+ 5'\).*12.*above the highest allowed value 9"):
        step(experiment.automaton, make_initial_grid(experiment, make_generator(0)), make_generator(0), 0)


def test_neighbours_layers_can_be_read(make_automaton, make_experiment):
    rules = [{"when": "self == FREE and sum(nb.fuel) >= 4", "then": "WALL"}]
    experiment = make_experiment(dict(LAYERED, rules=rules), initial={
        "type": "text", "rows": ["X.X.X"], "legend": {".": "FREE"}, "layers": {"fuel": {"type": "text", "rows": ["1 0 3 0 0"]}}})
    new = step(experiment.automaton, make_initial_grid(experiment, make_generator(0)), make_generator(0), 0)
    assert text(experiment.automaton.cells(new)) == "X#X.X"


@pytest.mark.parametrize("changes, where, message", [
    ({"update": "synchronous"}, "order", "only used by automata with"),
    ({"order": "sideways"}, "order", "must be one of: random, reading, reverse"),
    ({"otherwise": "FREE"}, "otherwise", "not used in agents mode"),
    ({"layers": {"self": {}}}, "layers.self", "reserved word"),
    ({"layers": {"fuel": {"min": 2}}}, "layers.fuel", "must include 0"),
    ({"layers": {"fuel": {"mx": 2}}}, "layers.fuel", "unknown field"),
    ({"rules": [{"table": {"X X X": "X"}}]}, "rules[0]", "tables are not available in agents mode"),
    ({"rules": [{"when": "self == X"}]}, "rules[0]", "needs 'then'"),
    ({"rules": [{"then": "X", "do": {"here": {"self": "X"}}}]}, "rules[0]", "either 'then' or 'do'"),
    ({"rules": [{"do": {}}]}, "rules[0].do", "needs 'here' and/or 'there'"),
    ({"rules": [{"do": {"there": {"self": "X"}}}]}, "rules[0].do.there", "needs 'to'"),
    ({"rules": [{"do": {"here": {"fuel": 1}}}]}, "rules[0].do.here.fuel", "is not something a cell holds (use: self)"),
    ({"rules": [{"do": {"to": {"among": ["north"]}, "here": {"self": "X"}}}]}, "rules[0].do.to.among", "'north' is not a direction (left, right)"),
    ({"rules": [{"do": {"to": {"choose": "best"}, "here": {"self": "X"}}}]}, "rules[0].do.to.choose", "must be one of: random, first"),
    ({"rules": [{"do": {"to": {"where": "nb.fuel > 0"}, "here": {"self": "X"}}}]}, "rules[0].do.to.where", "unknown layer 'fuel'"),
    ({"rules": [{"do": {"here": {"self": "nb"}}}]}, "rules[0].do.here.self", "'nb' can only be used inside"),
    ({"rules": [{"when": "direction == 0", "then": "X"}]}, "rules[0].when", "'direction' can only be used where a rule chooses"),
    ({"rules": [{"do": [{"here": {"self": "X"}, "wait": 1}]}]}, "rules[0].do[0].wait", "unknown field"),
])
def test_broken_agent_automata_are_rejected(make_automaton, changes, where, message):
    with pytest.raises(InputFileError) as error:
        make_automaton(MOVERS, **changes)
    assert error.value.location == where and message in error.value.message


def test_actions_need_the_agents_mode(make_automaton):
    with pytest.raises(InputFileError, match='actions need "update": "agents"'):
        make_automaton(rules=[{"do": {"here": {"self": "TREE"}}}])


@pytest.mark.parametrize("layers, where, message", [
    ({"speed": 1}, "initial.layers.speed", "has no layer 'speed' (its layers: fuel, trips)"),
    ({"fuel": "self +"}, "initial.layers.fuel", "not a valid expression"),
    ({"fuel": {"type": "text", "rows": ["1 2 3"]}}, "initial.layers.fuel", "3-cell grid, but the grid is 2-cell"),
    ({"fuel": {"type": "single", "value": 12, "background": 0}}, "initial.layers.fuel.value", "above the maximum 9"),
])
def test_broken_layer_starts_are_rejected(make_experiment, layers, where, message):
    with pytest.raises(InputFileError) as error:
        make_experiment(LAYERED, initial={"type": "text", "rows": ["X."], "legend": {".": "FREE"}, "layers": layers})
    assert error.value.location == where and message in error.value.message


# ------------------------------------------------------------ the brief's five

@pytest.mark.parametrize("steps", [1, 2, 11, 500])
def test_langtons_ant_matches_reference(steps):
    experiment = example_experiment("langtons_ant")
    automaton = experiment.automaton
    state = make_initial_grid(experiment, make_generator(0))
    for k in range(steps):
        state = step(automaton, state, make_generator(0), k)
    grid, position, heading = reference.langtons_ant(np.zeros((21, 21), int), (10, 10), 0, steps)
    assert automaton.cells(state).tolist() == grid
    assert np.argwhere(automaton.layer(state, "ant")).tolist() == [list(position)]
    assert automaton.layer(state, "heading")[position] == heading
    assert automaton.layer(state, "heading").sum() == heading           # nothing left behind


def test_langtons_ant_is_deterministic_and_reaches_its_highway():
    automaton = example_automaton("langtons_ant")
    assert automaton.deterministic and automaton.state_dtype == np.uint8
    # Once on the highway the ant repeats 104 steps that shift it 2 cells diagonally (a known result).
    grid, position, _ = reference.langtons_ant(np.zeros((120, 120), int), (60, 60), 0, 11000)
    later, moved, _ = reference.langtons_ant(grid, position, _, 104)
    assert sorted(abs(a - b) for a, b in zip(position, moved)) == [2, 2]


@pytest.mark.parametrize("seed", range(4))
def test_falling_sand_matches_reference(seed):
    automaton = example_automaton("falling_sand")
    assert automaton.deterministic
    rng = np.random.default_rng(seed)
    grids = run_steps(automaton, rng.choice(4, size=(9, 11), p=[0.5, 0.25, 0.1, 0.15]), 12)
    for k in range(12):
        assert grids[k + 1].tolist() == reference.falling_sand(grids[k].tolist()), f"step {k + 1} differs"
        assert np.bincount(grids[k + 1].ravel(), minlength=4).tolist() == np.bincount(grids[0].ravel(), minlength=4).tolist()


def test_falling_sand_example_settles_except_for_the_water(tmp_path):
    _, result = run_experiment(example_experiment("falling_sand"), tmp_path)
    answers = {a.question["ask"]: a.result for a in result.answers}
    assert answers["count_after"]["count"] == 22                           # no grain is lost
    assert answers["cycle"]["cycle"] is True and answers["cycle"]["length"] == 2
    assert answers["stable"]["stable"] is False


def test_schelling_keeps_everyone_and_ends_with_everyone_happy():
    experiment = example_experiment("schelling")
    automaton, rng = experiment.automaton, make_generator(7)
    state = start = make_initial_grid(experiment, rng)
    for k in range(60):
        state = step(automaton, state, rng, k)
    assert np.bincount(state.ravel()).tolist() == np.bincount(start.ravel()).tolist()
    assert (step(automaton, state, rng, 60) == state).all()                # nobody wants to move any more

    def same_neighbours(grid):
        around = [np.roll(grid, (dr, dc), (0, 1)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        return sum((n == grid) & (grid != 0) for n in around).sum() / sum((n != 0) & (grid != 0) for n in around).sum()
    assert same_neighbours(start) < 0.55 < 0.75 < same_neighbours(state)   # the groups have separated


def test_predator_prey_follows_its_rules():
    experiment = example_experiment("predator_prey")
    automaton, rng = experiment.automaton, make_generator(1)
    state = make_initial_grid(experiment, rng)
    rabbit, fox = automaton.states.parse("RABBIT"), automaton.states.parse("FOX")
    foxes = [(automaton.cells(state) == fox).sum()]
    for k in range(25):
        state = step(automaton, state, rng, k)
        cells, energy, age = automaton.cells(state), automaton.layer(state, "energy"), automaton.layer(state, "age")
        foxes.append((cells == fox).sum())
        assert (energy[cells == fox] >= 1).all() and (energy[cells != fox] == 0).all()
        assert (age[cells != rabbit] == 0).all() and age.max() <= automaton.parameters["breed_after"]
    assert all(later <= earlier for earlier, later in zip(foxes, foxes[1:]))   # foxes never multiply here
    assert (automaton.cells(state) == rabbit).sum() > 0


def test_ecosystem_keeps_all_three_alive_and_energy_only_on_animals():
    experiment = example_experiment("ecosystem")
    automaton, rng = experiment.automaton, make_generator(7)
    state = make_initial_grid(experiment, rng)
    names = automaton.states.names
    for k in range(120):
        state = step(automaton, state, rng, k)
    cells, energy = automaton.cells(state), automaton.layer(state, "energy")
    animal = (cells == names.index("RABBIT")) | (cells == names.index("FOX"))
    assert (energy[animal] >= 1).all() and (energy[~animal] == 0).all()
    assert energy.max() <= automaton.parameters["max_energy"]
    assert all((cells == value).sum() > 20 for value in (1, 2, 3))


# ------------------------------------------------------------ stored, resumed and fingerprinted like the rest

def small_ecosystem(make_experiment, **fields):
    import json
    from helpers import AUTOMATA
    automaton = json.loads((AUTOMATA / "ecosystem.json").read_text(encoding="utf-8"))
    settings = dict(size=[12, 14], steps=30, max_steps=30, checkpoint_every=7, seed=5,
                    initial={"type": "random", "weights": {"EMPTY": 3, "GRASS": 5, "RABBIT": 2, "FOX": 1},
                             "layers": {"energy": "6 * (self == RABBIT) + 9 * (self == FOX)"}},
                    questions=[{"ask": "count_after", "n": 30, "value": "RABBIT"}, {"ask": "state_after", "n": 12}])
    return make_experiment(automaton, **{**settings, **fields})


def folder_bytes(run) -> dict:
    return {name: run.folder.file(name).read_bytes() for name in (RunFolder.FINGERPRINTS, RunFolder.GENERATIONS, RunFolder.ANSWERS)}


def test_resumed_agents_run_equals_uninterrupted_run(make_experiment, tmp_path):
    straight, _ = run_experiment(small_ecosystem(make_experiment), tmp_path / "a")
    stopped = Run.start(small_ecosystem(make_experiment), runs_dir=tmp_path / "b")
    assert stopped.advance(stop_at=11).stopped
    resumed = Run.open(stopped.folder.path)
    assert resumed.state.step == 11 and resumed.state.grid.shape == (2, 12, 14)
    resumed.advance()
    assert folder_bytes(resumed) == folder_bytes(straight)
    assert straight.folder.records(RunFolder.FINGERPRINTS, 2 * FINGERPRINT_SIZE) == 31     # state + grid, per step


def test_same_seed_same_agents_run_and_layers_are_shown(make_experiment, tmp_path, capsys):
    from cellab.cli import main
    first, result = run_experiment(small_ecosystem(make_experiment), tmp_path / "a")
    again, _ = run_experiment(small_ecosystem(make_experiment), tmp_path / "b")
    other, _ = run_experiment(small_ecosystem(make_experiment, seed=6), tmp_path / "c")
    assert first.state.history.fingerprints == again.state.history.fingerprints != other.state.history.fingerprints
    rows = result.answers[1].result["grid"]
    assert len(rows) == 12 + 1 + 12 and rows[12] == "energy:"
    assert main(["show", str(first.folder.path), "--step", "12"]) == 0
    assert capsys.readouterr().out.splitlines()[1:] == rows
    assert main(["render", str(first.folder.path)]) == 0


def test_occurs_looks_at_the_grid_and_cycles_at_the_whole_state(make_experiment, tmp_path):
    # One thing bouncing between two walls: the grid "..X.." shows up on the way right and on the way
    # back, but with a different heading, so only the heading layer tells the two states apart.
    bouncer = dict(MOVERS, layers={"heading": {"max": 1}}, neighbourhood=[[1], [-1]], rules=[
        {"when": "self == X", "do": [
            {"to": {"where": "direction == heading and nb == FREE", "choose": "first"},
             "here": {"self": "FREE", "heading": 0}, "there": {"self": "X", "heading": "heading"}},
            {"here": {"heading": "1 - heading"}}]}])
    experiment = make_experiment(bouncer, steps=3, max_steps=50, initial={"type": "text", "rows": ["X...."], "legend": {".": "FREE"}},
                                 questions=[{"ask": "occurs", "rows": ["..X.."]}, {"ask": "occurs", "rows": ["X.X.."]},
                                            {"ask": "cycle"}, {"ask": "state_after", "n": 1007}])
    assert experiment.automaton.deterministic
    _, result = run_experiment(experiment, tmp_path)
    first, never, cycle, far = (answer.result for answer in result.answers)
    assert first == {"occurs": True, "first_step": 2}
    assert never == {"occurs": False, "certain": True}
    assert cycle == {"cycle": True, "start": 0, "length": 10}             # 4 steps right, turn, 4 back, turn
    assert far["grid"][0] == "..X.." and far["same_as_step"] == 7
