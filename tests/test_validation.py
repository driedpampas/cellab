"""Broken input files must be rejected before anything runs, with a message that
names the file, the field, and the problem."""

import json

import pytest

from cellab import load_automaton, load_experiment
from cellab.errors import InputFileError
from helpers import ELEMENTARY, FOREST_FIRE, write_json


def with_rule(when, then="FIRE"):
    return {"rules": [{"when": when, "then": then}]}


AUTOMATON_CASES = [
    # (changes to the forest fire file, field that must be named, words the message must contain)
    ({"neighborhood": "moore"}, "neighborhood", "unknown field"),
    ({"rules": None}, "rules", "valid list"),
    ({"dimensions": 3}, "dimensions", "must be 1 or 2"),
    (with_rule("self == FLAME"), "rules[0].when", "unknown name 'FLAME'"),
    (with_rule("self == FIER"), "rules[0].when", "unknown name 'FIER' (did you mean 'FIRE'?)"),
    (with_rule("self == Fire"), "rules[0].when", "unknown name 'Fire' (did you mean 'FIRE'?)"),
    (with_rule("__import__('os')"), "rules[0].when", "unknown function '__import__'"),
    (with_rule("self.real > 0"), "rules[0].when", "a dot is only used as nb.<layer>"),
    (with_rule("self == 'FIRE'"), "rules[0].when", "without quotes"),
    (with_rule("nb == FIRE"), "rules[0].when", "'nb' can only be used inside"),
    (with_rule("count(count(FIRE) > 1) > 0"), "rules[0].when", "cannot be used inside another"),
    (with_rule("(lambda: 1)()"), "rules[0].when", "only the built-in functions"),
    (with_rule("[1, 2]"), "rules[0].when", "List is not allowed"),
    (with_rule("self ==", "FIRE"), "rules[0].when", "not a valid expression"),
    (with_rule("random(1) < 0.5"), "rules[0].when", "random() takes no arguments"),
    (with_rule("min(nb, 1) > 0"), "rules[0].when", "'nb' can only be used inside"),
    ({"rules": [{"when": "self == TREE"}]}, "rules[0]", "needs 'then'"),
    ({"rules": [{"when": "True", "then": "FIRE", "table": {"1": "FIRE"}}]}, "rules[0]", "not both"),
    ({"states": ["EMPTY", "EMPTY"]}, "states", "unique"),
    ({"states": ["self", "TREE", "FIRE"]}, "states", "reserved word"),
    ({"states": ["EMPTY", "two words"]}, "states", "must be a word"),
    ({"states": []}, "states", "empty"),
    ({"states": {"type": "complex"}}, "states", "\"integer\" or \"real\""),
    ({"edges": {"type": "fixed"}}, "edges.value", "need a 'value'"),
    ({"edges": {"type": "fixed", "value": "WATER"}}, "edges.value", "'WATER' is not a state"),
    ({"edges": {"type": "wrap", "value": "EMPTY"}}, "edges.value", "take no value"),
    ({"edges": {"type": "mirror"}}, "edges.type", "\"wrap\" or \"fixed\""),
    ({"neighbourhood": "elementary"}, "neighbourhood", "1D neighbourhood"),
    ({"neighbourhood": "hexagonal"}, "neighbourhood", "unknown neighbourhood"),
    ({"neighbourhood": [[0, 0]]}, "neighbourhood", "not a neighbour"),
    ({"update": "sideways"}, "update", "unknown update mode"),
    ({"parameters": {"FIRE": 1}}, "parameters.FIRE", "already a state name"),
    ({"parameters": {"count": 1}}, "parameters.count", "reserved word"),
    ({"parameters": {"p": "high"}}, "parameters.p", "must be a number"),
    ({"colors": {"TREE": "green"}}, "colors.TREE", "colour like"),
    ({"colors": {"WATER": "#00ff00"}}, "colors.WATER", "not a state"),
]


@pytest.mark.parametrize("changes, where, words", AUTOMATON_CASES)
def test_broken_automaton_files_are_rejected(tmp_path, changes, where, words):
    data = {**FOREST_FIRE, **{k: v for k, v in changes.items() if v is not None}}
    if changes.get("rules", 1) is None:
        data["rules"] = "not a list"
    path = write_json(tmp_path / "broken.json", data)
    with pytest.raises(InputFileError) as caught:
        load_automaton(path)
    assert caught.value.location == where
    assert words in caught.value.message
    assert str(path) in str(caught.value)


@pytest.mark.parametrize("table, words", [
    ({"11": 0}, "has 2 cells"),
    ({"111": 0, "1 1 1": 1}, "listed twice"),
    ({"112": 0}, "above the maximum"),
    ({"111": 5}, "above the maximum"),
])
def test_broken_tables_are_rejected(tmp_path, table, words):
    data = dict(ELEMENTARY, rules=[{"table": table}])
    with pytest.raises(InputFileError) as caught:
        load_automaton(write_json(tmp_path / "table.json", data))
    assert caught.value.location == "rules[0].table"
    assert words in caught.value.message


def test_tables_need_bounded_states(tmp_path):
    data = dict(ELEMENTARY, states={"type": "integer", "min": 0}, rules=[{"table": {"111": 0}}])
    with pytest.raises(InputFileError, match="need both 'min' and 'max'"):
        load_automaton(write_json(tmp_path / "table.json", data))


def test_invalid_json_reports_line_and_column(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{\n  "name": "x",\n  "states": [1, 2,]\n}', encoding="utf-8")
    with pytest.raises(InputFileError) as caught:
        load_automaton(path)
    assert caught.value.location.startswith("line 3")
    assert "invalid JSON" in caught.value.message


def test_missing_file(tmp_path):
    with pytest.raises(InputFileError, match="file not found"):
        load_automaton(tmp_path / "nothing.json")


def test_files_saved_with_a_bom_are_read(tmp_path):
    path = tmp_path / "fire.json"
    path.write_text(json.dumps(FOREST_FIRE), encoding="utf-8-sig")
    assert load_automaton(path).states.names == tuple(FOREST_FIRE["states"])


EXPERIMENT_CASES = [
    ({"automaton": "missing.json"}, "automaton", "not found"),
    ({"initial": {"type": "random", "weights": {"TREE": 1}}}, "size", "is required"),
    ({"size": [3, 3]}, "size", "rows in 'initial'"),
    ({"size": [5]}, "size", "2D grid needs 2"),
    ({"initial": {"type": "text"}}, "initial.rows", "this field is required"),
    ({"initial": {"type": "drawing"}}, "initial.type", "must be one of"),
    ({"initial": {"type": "text", "rows": ["TX"], "legend": {"T": "TREE"}}}, "initial", "'X'"),
    ({"initial": {"type": "text", "rows": ["TT", "T"], "legend": {"T": "TREE"}}}, "initial", "same number"),
    ({"initial": {"type": "text", "rows": ["T"], "legend": {"T": "WATER"}}}, "initial", "'WATER' is not a state"),
    ({"steps": 50, "max_steps": 10}, "steps", "larger than max_steps"),
    ({"steps": -1}, "steps", "greater than or equal to 0"),
    ({"questions": [{"ask": "weather"}]}, "questions[0].ask", "must be one of"),
    ({"questions": [{"ask": "count_after", "n": 2, "value": "WATER"}]}, "questions[0].value", "not a state"),
    ({"questions": [{"ask": "state_after"}]}, "questions[0].n", "this field is required"),
    ({"questions": [{"ask": "occurs", "rows": ["T"], "legend": {"T": "TREE"}}]}, "questions[0].rows", "the configuration is"),
    ({"parameters": {"p": 0.5}}, "parameters.p", "has no parameter 'p'"),
    ({"seeds": 4}, "seeds", "unknown field"),
]


@pytest.mark.parametrize("changes, where, words", EXPERIMENT_CASES)
def test_broken_experiment_files_are_rejected(tmp_path, changes, where, words):
    write_json(tmp_path / "fire.json", FOREST_FIRE)
    data = {"name": "broken", "automaton": "fire.json", "steps": 3,
            "initial": {"type": "text", "rows": ["TF", "TT"], "legend": {"T": "TREE", "F": "FIRE"}}}
    data.update(changes)
    path = write_json(tmp_path / "experiment.json", data)
    with pytest.raises(InputFileError) as caught:
        load_experiment(path)
    assert caught.value.location == where
    assert words in caught.value.message


def test_errors_in_the_automaton_name_the_automaton_file(tmp_path):
    write_json(tmp_path / "fire.json", dict(FOREST_FIRE, rules=[{"when": "self == FLAME", "then": "FIRE"}]))
    path = write_json(tmp_path / "experiment.json", {
        "name": "x", "automaton": "fire.json", "steps": 1,
        "initial": {"type": "text", "rows": ["T"], "legend": {"T": "TREE"}}})
    with pytest.raises(InputFileError) as caught:
        load_experiment(path)
    assert caught.value.file.endswith("fire.json")
