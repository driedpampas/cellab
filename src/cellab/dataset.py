"""Datasets for machine learning: many short runs with their exact answers.

A dataset is described by a small file, like automata and experiments are:

    {
      "name": "elementary_w16",
      "automaton": "../automata/elementary.json",
      "size": [16],
      "settings": {"rule": {"range": [0, 255]}},        optional: parameter values to go through
      "initials": [{"type": "random", "weights": {"0": 1, "1": 1}}, ...],
      "samples_per_setting": 200,
      "horizons": [1, 2, 4, 8, 16, 32, 64],             the N of "after N steps"
      "max_steps": 70000,                               how far a run may go to find its cycle
      "occurs_queries": 2,                              "does C occur?" questions per run
      "seed": 2026
    }

Every sample is one run of the ordinary engine from its own seed (seed of the
dataset * 1,000,003 + sample number), so any sample can be reproduced alone, and
the labels are the same facts `cellab run` reports: the grid and the counts after
N steps, where the cycle starts and how long it is, from which step the grid is
stable, and whether a configuration occurs.

Output: <name>.npz (arrays, one row per sample) and <name>.json (what they mean).
`question_answer_pairs` turns the same data into questions and answers in words.
"""

import gzip
import itertools
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .automaton import load_automaton
from .engine import step
from .errors import InputFileError
from .experiment import Experiment, load_experiment, make_initial_grid
from .fingerprints import History, fingerprint
from .grids import format_grid
from .loading import read_json, validate
from .randomness import make_generator

SEED_STRIDE = 1_000_003        # sample seed = dataset seed * SEED_STRIDE + sample number
QUERY_SEED_OFFSET = 10**15     # the queries of a sample are drawn with seed + this
UNKNOWN = -1                   # "the run did not settle this" in every label array

FIELDS = {
    "setting": "which entry of 'settings' (in the .json file) the sample uses",
    "seed": "the sample's own seed: rerun it with this seed to get exactly this sample",
    "initial": "generation 0 (for automata with layers: the grid first, then the layers)",
    "states": "the grid after each horizon N: states[sample, horizon number]",
    "counts": "how many cells hold each state value after each horizon: counts[sample, horizon number, value]",
    "steps_run": "how many steps the run was simulated",
    "cycle_start": "the first step of the cycle the run enters, or -1 if none was found within max_steps",
    "cycle_length": "the length of that cycle (1 means the grid is stable), or -1",
    "stable_from": "the first step from which the grid did not change any more, or -1",
    "queries": "configurations C for the question 'does C ever occur?': queries[sample, query number]",
    "occurs": "the answer for each query: 1 yes, 0 never (certain), -1 not seen within the run but not certain",
    "occurs_step": "the first step at which the query occurs, or -1",
}


class _DatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    automaton: str
    size: list[int]
    settings: dict[str, Any] = Field(default_factory=dict)
    initials: list[dict[str, Any]] = Field(min_length=1)
    samples_per_setting: int = Field(ge=1)
    horizons: list[int] = Field(min_length=1)
    max_steps: int = Field(ge=1)
    occurs_queries: int = Field(default=2, ge=0)
    seed: int = Field(default=0, ge=0)


def load_spec(path: str | Path) -> tuple[_DatasetFile, list[dict]]:
    """Read a dataset file. Returns it and the list of settings (parameter values) it goes through."""
    path = Path(path)
    spec = validate(_DatasetFile, read_json(path), path)

    def fail(where: str, message: str):
        raise InputFileError(path, where, message)

    if not (path.parent / spec.automaton).is_file():
        fail("automaton", f"file '{spec.automaton}' not found (paths are relative to the dataset file)")
    if sorted(set(spec.horizons)) != spec.horizons or spec.horizons[0] < 1:
        fail("horizons", "must be increasing whole numbers from 1 up")
    if spec.horizons[-1] > spec.max_steps:
        fail("horizons", f"the largest is beyond max_steps ({spec.max_steps})")
    values = {}
    for name, raw in spec.settings.items():
        if isinstance(raw, dict) and list(raw) == ["range"] and len(raw["range"]) == 2:
            values[name] = list(range(raw["range"][0], raw["range"][1] + 1))
        elif isinstance(raw, list) and raw:
            values[name] = raw
        else:
            fail(f"settings.{name}", "must be a list of values or {\"range\": [first, last]}")
    settings = [dict(zip(values, combination)) for combination in itertools.product(*values.values())]
    return spec, settings


# ------------------------------------------------------------ one sample

def run_sample(experiment: Experiment, seed: int, horizons: list[int], max_steps: int, queries: int) -> dict:
    """Simulate one run and collect its labels (one entry per name in FIELDS, except "setting")."""
    automaton = experiment.automaton
    rng = make_generator(seed)
    state = make_initial_grid(experiment, rng)
    sample = {"seed": seed, "initial": state}
    history, grids = History(), []
    while True:
        cells = automaton.cells(state)
        grids.append(cells)
        history.record(fingerprint(state), fingerprint(cells) if automaton.layers else None)
        k = history.last_step
        settled = history.repeat is not None if automaton.deterministic else True
        if k >= horizons[-1] and (settled or k >= max_steps):
            break
        state = step(automaton, state, rng, k)

    sample["states"] = np.stack([grids[n] for n in horizons])
    values = _values(automaton.states)
    if values is not None:
        sample["counts"] = np.array([[np.count_nonzero(grids[n] == v) for v in values] for n in horizons], dtype=np.int32)
    sample["steps_run"] = history.last_step
    certain = automaton.deterministic and history.cycle is not None      # the whole future is known
    sample["cycle_start"], sample["cycle_length"] = history.cycle if certain else (UNKNOWN, UNKNOWN)
    stable = history.standstill is not None and (certain or not automaton.deterministic)
    sample["stable_from"] = history.standstill - 1 if stable else UNKNOWN

    # "Does C occur?": half the queries are grids of this run, half are such grids with one or
    # two cells changed (usually absent, and close enough to the real ones to be hard to tell).
    draw = make_generator(seed + QUERY_SEED_OFFSET)   # its own generator: the run's draws stay untouched
    sample["queries"] = np.empty((queries, *cells.shape), dtype=cells.dtype)
    sample["occurs"] = np.empty(queries, dtype=np.int8)
    sample["occurs_step"] = np.empty(queries, dtype=np.int32)
    for q in range(queries):
        query = grids[draw.integers(len(grids))].copy()
        if q % 2 == 1 and values is not None and len(values) > 1:
            for _ in range(draw.integers(1, 3)):
                position = tuple(draw.integers(n) for n in query.shape)
                query[position] = draw.choice([v for v in values if v != query[position]])
        first = history.grid_seen.get(fingerprint(query), UNKNOWN)
        sample["queries"][q] = query
        sample["occurs_step"][q] = first
        sample["occurs"][q] = 1 if first != UNKNOWN else (0 if certain else UNKNOWN)
    return sample


def _values(states) -> list[int] | None:
    """The values a cell can take, if they can be listed (named states, or whole numbers with both limits)."""
    if states.kind == "real" or states.highest is None or states.lowest is None:
        return None
    return list(range(int(states.lowest), int(states.highest) + 1))


# ------------------------------------------------------------ a whole dataset

def _run_setting(job: tuple) -> list[dict]:
    """All samples of one setting (run in a worker process)."""
    spec_path, number, setting = job
    spec, _ = load_spec(spec_path)
    automaton_path = (Path(spec_path).parent / spec.automaton).resolve()
    samples = []
    with tempfile.TemporaryDirectory() as folder:
        experiments = []
        for i, initial in enumerate(spec.initials):
            file = Path(folder) / f"experiment_{i}.json"
            file.write_text(json.dumps({"name": spec.name, "automaton": str(automaton_path), "size": spec.size,
                                        "initial": initial, "steps": 0, "parameters": setting}), encoding="utf-8")
            experiments.append(load_experiment(file))
        for k in range(spec.samples_per_setting):
            index = number * spec.samples_per_setting + k
            sample = run_sample(experiments[k % len(experiments)], spec.seed * SEED_STRIDE + index,
                                spec.horizons, spec.max_steps, spec.occurs_queries)
            samples.append({"setting": number, **sample})
    return samples


def build_dataset(spec_path: str | Path, out_dir: str | Path = "data", workers: int = 1,
                  on_progress=None) -> Path:
    """Generate the dataset a file describes. Returns the path of the .npz file written."""
    spec_path = Path(spec_path).resolve()
    spec, settings = load_spec(spec_path)
    settings = settings or [{}]
    jobs = [(spec_path, number, setting) for number, setting in enumerate(settings)]
    samples: list[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(_run_setting, jobs)
            for done, result in enumerate(results, start=1):
                samples += result
                if on_progress:
                    on_progress(done, len(jobs))
    else:
        for done, job in enumerate(jobs, start=1):
            samples += _run_setting(job)
            if on_progress:
                on_progress(done, len(jobs))

    arrays = {name: np.array([sample[name] for sample in samples]) for name in samples[0]}
    arrays["setting"] = arrays["setting"].astype(np.int32)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{spec.name}.npz", **arrays)

    from . import __version__
    automaton = load_automaton(spec_path.parent / spec.automaton)
    meta = {
        "name": spec.name, "description": spec.description, "made_by": f"cellab {__version__}",
        "samples": len(samples), "spec": spec.model_dump(), "settings": settings,
        "automaton": {"name": automaton.name, "file": Path(spec.automaton).name, "sha256": automaton.digest,
                      "dimensions": automaton.dimensions, "deterministic": automaton.deterministic,
                      "state_names": list(automaton.states.names), "values": _values(automaton.states),
                      "layers": list(automaton.layers), "edges": automaton.edge_mode},
        "seed_of_sample": f"{spec.seed} * {SEED_STRIDE} + sample number",
        "arrays": {name: {"shape": list(array.shape), "type": str(array.dtype), "meaning": FIELDS[name]}
                   for name, array in arrays.items()},
    }
    (out / f"{spec.name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out / f"{spec.name}.npz"


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a dataset back: (arrays by name, the description from the .json file)."""
    path = Path(path)
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


# ------------------------------------------------------------ questions and answers in words

def question_answer_pairs(arrays: dict[str, np.ndarray], meta: dict, limit: int | None = None) -> Iterator[dict]:
    """The dataset as questions and answers in words, sample by sample (the five questions of the brief)."""
    from .states import StateSpace
    info, spec = meta["automaton"], meta["spec"]
    names = tuple(info["state_names"])
    numbers = "real" if arrays["states"].dtype.kind == "f" else "integer"
    states = StateSpace(kind="named", names=names) if names else StateSpace(kind=numbers)
    layers = len(info["layers"])

    def grid_text(grid) -> str:
        return " / ".join(format_grid(grid, states))

    count = len(arrays["seed"]) if limit is None else min(limit, len(arrays["seed"]))
    for i in range(count):
        setting = meta["settings"][arrays["setting"][i]]
        initial = arrays["initial"][i]
        context = {"sample": i, "automaton": info["name"], "parameters": setting, "seed": int(arrays["seed"][i]),
                   "initial": grid_text(initial[0] if layers else initial)}

        def pair(ask: str, question: str, answer: str, **more):
            return {**context, "ask": ask, "question": question, "answer": answer, **more}

        for h, n in enumerate(spec["horizons"]):
            steps = "1 step" if n == 1 else f"{n} steps"
            yield pair("state_after", f"What is the state after {steps}?", grid_text(arrays["states"][i, h]), n=n)
            if "counts" in arrays:
                for value, total in zip(info["values"], arrays["counts"][i, h]):
                    label = states.format(value)
                    yield pair("count_after", f"How many cells are {label} after {steps}?", str(int(total)), n=n, value=label)

        start, length, stable = (int(arrays[name][i]) for name in ("cycle_start", "cycle_length", "stable_from"))
        limit_text = f"not within the {int(arrays['steps_run'][i])} steps simulated"
        if stable != UNKNOWN:
            answer = f"From step {stable}." if info["deterministic"] else f"It stopped changing at step {stable} (the rules are random, so this is not certain)."
        elif length != UNKNOWN:
            answer = f"Never: from step {start} it repeats in a cycle of length {length}."
        else:
            answer = f"Unknown: {limit_text}."
        yield pair("stable", "When does it reach a stable state?", answer)
        answer = f"Yes: from step {start}, with a cycle of length {length}." if length != UNKNOWN else f"Unknown: {limit_text}."
        yield pair("cycle", "Does it enter a cycle?", answer)
        for q in range(arrays["queries"].shape[1]):
            occurs, first = int(arrays["occurs"][i, q]), int(arrays["occurs_step"][i, q])
            answer = f"Yes, first at step {first}." if occurs == 1 else "No, never." if occurs == 0 else f"Unknown: {limit_text}."
            yield pair("occurs", f"Does the configuration {grid_text(arrays['queries'][i, q])} ever occur?", answer)


def write_question_answer_pairs(dataset: str | Path, out: str | Path, limit: int | None = None) -> int:
    """Write the questions and answers as JSON lines (compressed if `out` ends in .gz). Returns how many."""
    arrays, meta = load_dataset(dataset)
    opener = gzip.open if str(out).endswith(".gz") else open
    written = 0
    with opener(out, "wt", encoding="utf-8") as handle:
        for pair in question_answer_pairs(arrays, meta, limit):
            handle.write(json.dumps(pair) + "\n")
            written += 1
    return written
