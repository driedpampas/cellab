"""Reading an experiment file: what to do with an automaton.

An experiment names an automaton file and gives everything a run needs:

    size, initial         the grid and how the first generation is made
                          (for automata with layers: also "initial.layers")
    steps                 how many generations to simulate (and show/store)
    seed                  for reproducible randomness (chosen and recorded if missing)
    max_steps             how far a run may go to answer the questions
    store_generations     keep every generation on disk (for pictures, replays)
    checkpoint_every      how often the run's state is saved for resuming
    parameters            optional new values for the automaton's parameters
    questions             the five questions and their parameters

Like automaton.py, loading checks the structure first (pydantic) and the meaning
second, so problems are reported before a run starts.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .automaton import Automaton, load_automaton
from .errors import InputFileError
from .expressions import EvalContext, Expression, compile_expression
from .grids import format_grid, parse_grid, parse_legend
from .loading import read_json, validate
from .questions import Cycle, CountAfter, Occurs, Question, Stable, StateAfter
from .states import StateSpace, checked_values

MAX_CELLS = 10_000_000


# ------------------------------------------------------------ file structure

class _ExperimentFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    automaton: str
    size: list[int] | None = None
    initial: dict[str, Any]
    steps: int = Field(ge=0)
    seed: int | None = Field(default=None, ge=0)
    max_steps: int = Field(default=10_000, ge=1)
    store_generations: bool = True
    checkpoint_every: int = Field(default=100, ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    questions: list[dict[str, Any]] = Field(default_factory=list)


class _Start(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    layers: dict[str, Any] = Field(default_factory=dict)   # how each layer starts (default: all 0)


class _TextStart(_Start):
    rows: list[str] = Field(min_length=1)
    legend: dict[str, Any] = Field(default_factory=dict)


class _RandomStart(_Start):
    weights: dict[str, float] = Field(min_length=1)


class _SingleStart(_Start):
    value: Any
    background: Any
    position: list[int] | None = None   # default: the centre


_STARTS = {"text": _TextStart, "random": _RandomStart, "single": _SingleStart}


class _NQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ask: str
    n: int = Field(ge=0)


class _CountQuestion(_NQuestion):
    value: Any


class _PlainQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ask: str


class _OccursQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ask: str
    rows: list[str] = Field(min_length=1)
    legend: dict[str, Any] = Field(default_factory=dict)


_QUESTIONS = {"state_after": _NQuestion, "count_after": _CountQuestion, "stable": _PlainQuestion,
              "cycle": _PlainQuestion, "occurs": _OccursQuestion}


# ------------------------------------------------------------ the result

@dataclass(frozen=True, eq=False)
class Experiment:
    name: str
    automaton: Automaton
    shape: tuple[int, ...]
    start: BaseModel                      # one of the _...Start models above
    start_grid: np.ndarray | None         # the parsed grid for "text" starts
    layer_starts: dict[str, Any]          # layer name -> (start, parsed grid) or an Expression of the grid
    legend: dict[str, int | float]        # symbols for reading and printing grids
    steps: int
    seed: int | None
    max_steps: int
    store_generations: bool
    checkpoint_every: int
    questions: tuple[Question, ...]
    source: Path
    raw: dict                             # the file's content, copied into run folders

    def with_steps(self, steps: int) -> "Experiment":
        raw = dict(self.raw, steps=steps)
        return dataclasses.replace(self, steps=steps, max_steps=max(self.max_steps, steps), raw=raw)

    def format(self, state: np.ndarray) -> list[str]:
        """A generation as text rows, with this experiment's symbols (layers follow the grid)."""
        automaton = self.automaton
        rows = format_grid(automaton.cells(state), automaton.states, self.legend)
        for name, space in automaton.layers.items():
            rows += [f"{name}:"] + format_grid(automaton.layer(state, name), space)
        return rows


def load_experiment(path: str | Path) -> Experiment:
    """Read and check an experiment file, and the automaton file it names."""
    path = Path(path)
    data = read_json(path)
    model = validate(_ExperimentFile, data, path)

    def fail(where: str, message: str):
        raise InputFileError(path, where, message)

    automaton_path = path.parent / model.automaton
    if not automaton_path.is_file():
        fail("automaton", f"file '{model.automaton}' not found (paths are relative to the experiment file)")
    automaton = load_automaton(automaton_path)
    automaton = _override_parameters(automaton, model.parameters, fail)
    states, dims = automaton.states, automaton.dimensions

    # The initial world --------------------------------------------------
    start, start_grid, legend = _read_start(model.initial, states, dims, path, "initial", fail)
    shape = _resolve_shape(model.size, start_grid, dims, fail)
    _check_start(start, states, shape, fail, "initial")

    layer_starts: dict[str, Any] = {}
    for name, raw in start.layers.items():
        where = f"initial.layers.{name}"
        if name not in automaton.layers:
            known = ", ".join(automaton.layers) or "none"
            fail(where, f"the automaton has no layer '{name}' (its layers: {known})")
        if isinstance(raw, dict):                 # written like "initial" itself: text, random or single
            layer_start, layer_grid, _ = _read_start(raw, automaton.layers[name], dims, path, where, fail)
            if layer_start.layers:
                fail(f"{where}.layers", "a layer has no layers of its own")
            if layer_grid is not None and layer_grid.shape != shape:
                fail(where, f"the rows make a {_size_text(layer_grid.shape)} grid, but the grid is {_size_text(shape)}")
            _check_start(layer_start, automaton.layers[name], shape, fail, where)
            layer_starts[name] = (layer_start, layer_grid)
        else:                                     # a value computed from the initial grid
            layer_starts[name] = compile_expression(raw, where, str(path), frozenset(automaton.constants) | {"self"},
                                                    frozenset(states.names))

    # Run settings -------------------------------------------------------
    if model.steps > model.max_steps:
        fail("steps", f"is larger than max_steps ({model.max_steps})")

    # Questions ----------------------------------------------------------
    questions = []
    for index, raw in enumerate(model.questions):
        where = f"questions[{index}]"
        ask = raw.get("ask")
        if ask not in _QUESTIONS:
            fail(f"{where}.ask", f"must be one of: {', '.join(_QUESTIONS)}")
        q = validate(_QUESTIONS[ask], raw, path, prefix=where)
        if ask == "state_after":
            questions.append(StateAfter(n=q.n))
        elif ask == "count_after":
            try:
                value = states.parse(q.value)
            except ValueError as error:
                fail(f"{where}.value", str(error))
            questions.append(CountAfter(n=q.n, value=value, value_text=states.format(value)))
        elif ask == "stable":
            questions.append(Stable())
        elif ask == "cycle":
            questions.append(Cycle())
        else:
            try:
                own_legend = parse_legend(q.legend, states) if q.legend else legend
                target = parse_grid(q.rows, states, dims, own_legend)
            except ValueError as error:
                fail(where, str(error))
            if target.shape != shape:
                fail(f"{where}.rows", f"the configuration is {_size_text(target.shape)}, "
                                      f"but the grid is {_size_text(shape)}")
            questions.append(Occurs(grid=target, rows=format_grid(target, states, legend)))

    return Experiment(
        name=model.name, automaton=automaton, shape=shape, start=start, start_grid=start_grid,
        layer_starts=layer_starts, legend=legend, steps=model.steps, seed=model.seed, max_steps=model.max_steps,
        store_generations=model.store_generations, checkpoint_every=model.checkpoint_every,
        questions=tuple(questions), source=path, raw=data,
    )


def make_initial_grid(experiment: Experiment, rng: np.random.Generator) -> np.ndarray:
    """Build generation 0: the grid and, for automata with layers, the layers after it.
    Random starts draw from the run's generator (the first draws of the run)."""
    automaton, shape = experiment.automaton, experiment.shape
    grid = _start_grid(experiment.start, experiment.start_grid, automaton.states, shape, rng)
    if not automaton.layers:
        return grid
    state = np.zeros(automaton.state_shape(shape), dtype=automaton.state_dtype)
    state[0] = grid
    for number, (name, space) in enumerate(automaton.layers.items(), start=1):
        spec = experiment.layer_starts.get(name)
        if isinstance(spec, Expression):
            ctx = EvalContext(state[0], None, (), automaton.dimensions, automaton.constants,
                              automaton.states.constants(), rng, 0)
            state[number] = space.finish_step(checked_values(spec.evaluate(ctx), space.kind, spec.where))
        elif spec is not None:
            state[number] = _start_grid(*spec, space, shape, rng)
    return state


def _start_grid(start, start_grid, space: StateSpace, shape, rng: np.random.Generator) -> np.ndarray:
    if isinstance(start, _TextStart):
        return start_grid.copy()
    if isinstance(start, _RandomStart):
        values = [space.parse(token) for token in start.weights]
        weights = np.array(list(start.weights.values()), dtype=float)
        chosen = rng.choice(len(values), size=shape, p=weights / weights.sum())
        return np.array(values, dtype=space.dtype)[chosen]
    grid = np.full(shape, space.parse(start.background), dtype=space.dtype)
    position = tuple(start.position) if start.position is not None else tuple(n // 2 for n in shape)
    grid[position] = space.parse(start.value)
    return grid


# ------------------------------------------------------------ helpers

def _read_start(raw: dict, space: StateSpace, dims: int, path: Path, where: str, fail):
    """Check the structure of an initial world (of the grid or of a layer).
    Returns (the start model, the parsed grid of a "text" start or None, the legend)."""
    kind = raw.get("type")
    if kind not in _STARTS:
        fail(f"{where}.type", f"must be one of: {', '.join(_STARTS)}")
    start = validate(_STARTS[kind], raw, path, prefix=where)
    legend, grid = {}, None
    if isinstance(start, _TextStart):
        try:
            legend = parse_legend(start.legend, space)
            grid = parse_grid(start.rows, space, dims, legend)
        except ValueError as error:
            fail(where, str(error))
    return start, grid, legend


def _override_parameters(automaton: Automaton, overrides: dict, fail) -> Automaton:
    if not overrides:
        return automaton
    parameters = dict(automaton.parameters)
    for name, value in overrides.items():
        if name not in parameters:
            known = ", ".join(parameters) or "none"
            fail(f"parameters.{name}", f"the automaton has no parameter '{name}' (its parameters: {known})")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(f"parameters.{name}", "must be a number")
        parameters[name] = value
    constants = {**automaton.constants, **parameters}
    return dataclasses.replace(automaton, parameters=parameters, constants=constants)


def _resolve_shape(size, start_grid, dims, fail) -> tuple[int, ...]:
    if size is not None:
        if len(size) != dims:
            fail("size", f"a {dims}D grid needs {dims} number(s): " + ("[cells]" if dims == 1 else "[rows, columns]"))
        if any(n < 1 for n in size):
            fail("size", "every size must be at least 1")
        shape = tuple(size)
        if start_grid is not None and start_grid.shape != shape:
            fail("size", f"is {_size_text(shape)}, but the rows in 'initial' make a {_size_text(start_grid.shape)} grid")
    elif start_grid is not None:
        shape = start_grid.shape
    else:
        fail("size", "is required unless 'initial' gives the rows of the grid")
    if int(np.prod(shape)) > MAX_CELLS:
        fail("size", f"the grid has more than {MAX_CELLS:,} cells")
    return shape


def _check_start(start, states, shape, fail, where: str) -> None:
    if isinstance(start, _RandomStart):
        for token, weight in start.weights.items():
            try:
                states.parse(token)
            except ValueError as error:
                fail(f"{where}.weights.{token}", str(error))
            if weight < 0:
                fail(f"{where}.weights.{token}", "weights cannot be negative")
        if sum(start.weights.values()) <= 0:
            fail(f"{where}.weights", "at least one weight must be above 0")
    elif isinstance(start, _SingleStart):
        for label in ("value", "background"):
            try:
                states.parse(getattr(start, label))
            except ValueError as error:
                fail(f"{where}.{label}", str(error))
        if start.position is not None:
            if len(start.position) != len(shape) or not all(0 <= p < n for p, n in zip(start.position, shape)):
                fail(f"{where}.position", f"must be inside the {_size_text(shape)} grid (counting from 0)")


def _size_text(shape: tuple[int, ...]) -> str:
    return f"{shape[0]}-cell" if len(shape) == 1 else f"{shape[0]} x {shape[1]}"
