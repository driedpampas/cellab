"""Datasets: every label must be exactly what the ordinary engine and its questions report."""

import gzip
import json

import numpy as np
import pytest

from cellab import InputFileError
from cellab.cli import main
from cellab.dataset import UNKNOWN, build_dataset, load_dataset, load_spec, question_answer_pairs
from helpers import AUTOMATA, ELEMENTARY, ROOT, answer_results, run_experiment, write_json


def spec_file(tmp_path, **changes):
    data = {"name": "tiny", "automaton": str(AUTOMATA / "elementary.json"), "size": [10],
            "settings": {"rule": [30, 90, 184, 250]},
            "initials": [{"type": "random", "weights": {"0": 1, "1": 1}}, {"type": "random", "weights": {"0": 3, "1": 1}}],
            "samples_per_setting": 6, "horizons": [1, 2, 5, 20], "max_steps": 2000, "occurs_queries": 4, "seed": 3}
    return write_json(tmp_path / "tiny.json", {**data, **changes})


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dataset")
    return load_dataset(build_dataset(spec_file(folder), folder))


def test_shapes_and_description(tiny):
    arrays, meta = tiny
    assert arrays["initial"].shape == (24, 10) and arrays["states"].shape == (24, 4, 10)
    assert arrays["counts"].shape == (24, 4, 2) and arrays["queries"].shape == (24, 4, 10)
    assert (arrays["counts"].sum(axis=2) == 10).all()
    assert meta["samples"] == 24 and meta["settings"][2] == {"rule": 184}
    assert set(meta["arrays"]) == set(arrays) and all(entry["meaning"] for entry in meta["arrays"].values())
    assert arrays["setting"].tolist() == [k // 6 for k in range(24)]


def test_labels_equal_the_answers_of_an_ordinary_run(tiny, make_experiment, tmp_path):
    arrays, meta = tiny
    for i in (0, 7, 13, 22):
        rule = meta["settings"][arrays["setting"][i]]["rule"]
        row = "".join(map(str, arrays["initial"][i]))
        query = "".join(map(str, arrays["queries"][i, 1]))
        experiment = make_experiment(
            automaton=dict(ELEMENTARY, parameters={"rule": rule}), steps=20, max_steps=2000, initial={"type": "text", "rows": [row]},
            questions=[{"ask": "state_after", "n": 5}, {"ask": "count_after", "n": 20, "value": 1}, {"ask": "cycle"},
                       {"ask": "stable"}, {"ask": "occurs", "rows": [query]}])
        _, result = run_experiment(experiment, tmp_path / str(i))
        answers = answer_results(result)
        assert answers["state_after"]["grid"] == ["".join(map(str, arrays["states"][i, 2]))]
        assert answers["count_after"]["count"] == arrays["counts"][i, 3, 1]
        assert (answers["cycle"]["start"], answers["cycle"]["length"]) == (arrays["cycle_start"][i], arrays["cycle_length"][i])
        assert answers["stable"].get("from_step", UNKNOWN) == arrays["stable_from"][i]
        assert answers["stable"]["stable"] == (arrays["cycle_length"][i] == 1)
        assert answers["occurs"]["occurs"] == bool(arrays["occurs"][i, 1])
        assert answers["occurs"].get("first_step", UNKNOWN) == arrays["occurs_step"][i, 1]


def test_queries_include_both_answers_and_real_grids_always_occur(tiny):
    arrays, _ = tiny
    assert (arrays["occurs"][:, 0::2] == 1).all()                       # taken from the run itself
    assert set(np.unique(arrays["occurs"][:, 1::2])) == {0, 1}           # changed grids: mostly absent
    assert ((arrays["occurs"] == 1) == (arrays["occurs_step"] >= 0)).all()


def test_same_file_same_dataset_and_any_number_of_workers(tmp_path, tiny):
    again = load_dataset(build_dataset(spec_file(tmp_path), tmp_path / "again", workers=2))[0]
    assert all((again[name] == array).all() for name, array in tiny[0].items())
    other = load_dataset(build_dataset(spec_file(tmp_path, seed=4), tmp_path / "other"))[0]
    assert (other["initial"] != tiny[0]["initial"]).any()


def test_runs_that_do_not_settle_are_marked_unknown(tmp_path):
    arrays, _ = load_dataset(build_dataset(spec_file(tmp_path, settings={"rule": [30]}, size=[40], max_steps=30), tmp_path))
    assert (arrays["cycle_length"] == UNKNOWN).all() and (arrays["stable_from"] == UNKNOWN).all()
    assert (arrays["steps_run"] == 30).all()
    assert set(np.unique(arrays["occurs"][:, 1::2])) <= {UNKNOWN, 1}     # "not seen" is not "never"


def test_automata_with_layers_and_random_rules(tmp_path):
    spec = spec_file(tmp_path, automaton=str(AUTOMATA / "ecosystem.json"), size=[8, 9], settings={}, samples_per_setting=4,
                     horizons=[1, 3], max_steps=3, initials=[{"type": "random", "weights": {"EMPTY": 3, "GRASS": 5, "RABBIT": 2, "FOX": 1},
                                                              "layers": {"energy": "6 * (self != EMPTY and self != GRASS)"}}])
    arrays, meta = load_dataset(build_dataset(spec, tmp_path))
    assert arrays["initial"].shape == (4, 2, 8, 9) and arrays["states"].shape == (4, 2, 8, 9)
    assert meta["automaton"]["layers"] == ["energy"] and not meta["automaton"]["deterministic"]
    assert (arrays["cycle_length"] == UNKNOWN).all() and (arrays["occurs"] != 0).all()


def test_questions_and_answers_in_words(tiny, tmp_path, capsys):
    arrays, meta = tiny
    pairs = list(question_answer_pairs(arrays, meta, limit=2))
    per_sample = 4 * (1 + 2) + 1 + 1 + 4                                # horizons x (state + 2 counts), stable, cycle, queries
    assert len(pairs) == 2 * per_sample
    first = pairs[0]
    assert first["automaton"] == "Elementary rule" or first["automaton"]
    assert first["question"] == "What is the state after 1 step?" and first["parameters"] == {"rule": 30}
    assert first["answer"] == "".join(map(str, arrays["states"][0, 0]))
    assert {pair["ask"] for pair in pairs} == {"state_after", "count_after", "stable", "cycle", "occurs"}
    cycle = next(pair for pair in pairs if pair["ask"] == "cycle")
    assert cycle["answer"] == f"Yes: from step {arrays['cycle_start'][0]}, with a cycle of length {arrays['cycle_length'][0]}."

    spec = spec_file(tmp_path)
    assert main(["dataset", str(spec), "--out", str(tmp_path / "d")]) == 0
    assert main(["qa", str(tmp_path / "d" / "tiny.npz"), "--limit", "3"]) == 0
    with gzip.open(tmp_path / "d" / "tiny_qa.jsonl.gz", "rt", encoding="utf-8") as handle:
        assert len([json.loads(line) for line in handle]) == 3 * per_sample


@pytest.mark.parametrize("changes, where, message", [
    ({"automaton": "nowhere.json"}, "automaton", "not found"),
    ({"horizons": [2, 1]}, "horizons", "increasing"),
    ({"horizons": [1, 5000]}, "horizons", "beyond max_steps"),
    ({"settings": {"rule": 30}}, "settings.rule", "must be a list of values"),
    ({"samples": 5}, "samples", "unknown field"),
])
def test_broken_dataset_files_are_rejected(tmp_path, changes, where, message):
    with pytest.raises(InputFileError) as error:
        load_spec(spec_file(tmp_path, **changes))
    assert error.value.location == where and message in error.value.message


@pytest.mark.parametrize("path", sorted((ROOT / "datasets").glob("*.json")), ids=lambda p: p.stem)
def test_every_dataset_file_is_valid(path):
    spec, settings = load_spec(path)
    assert spec.name == path.stem and spec.horizons[0] == 1
