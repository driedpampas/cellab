"""Reading an automaton file and turning it into a checked, ready-to-run Automaton.

Loading happens in three layers, each with its own kind of error message:

1. JSON          is the file valid JSON?                         (loading.read_json)
2. structure     are the fields there, spelled right, of the     (pydantic model below)
                 right basic type?
3. meaning       do the states, neighbourhood, edges, parameters (build_automaton)
                 and rules make sense together? Every rule is
                 compiled here, so an unknown name or a typo
                 stops the program before anything runs.

The result, `Automaton`, contains no file text any more: states are numbers,
the neighbourhood is a list of offsets and every rule is a compiled object.

Two update modes exist. "synchronous" (the default): every cell computes its own
new value from the old grid (engine.py). "agents": cells can also change another
cell, which is what moving, eating and multiplying need (agents.py). Only agents
automata may declare "layers" (extra whole numbers per cell) and an "order".
"""

import hashlib
import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .agents import ORDERS, AgentRule, build_agent_rule
from .errors import InputFileError
from .expressions import RESERVED_NAMES, Expression, compile_expression
from .loading import read_json, validate
from .neighbourhood import DIRECTIONS, resolve_neighbourhood
from .rules import ExpressionRule, TableRule, build_table_rule
from .states import StateSpace, layer_space_from_file, state_space_from_file

UPDATE_MODES = ("synchronous", "agents")


# ------------------------------------------------------------ file structure

class _Edges(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    value: Any = None


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    when: Any = None
    then: Any = None
    table: dict[str, Any] | None = None
    do: Any = None


class _AutomatonFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    dimensions: int
    states: Any
    neighbourhood: Any
    edges: _Edges
    update: str = "synchronous"
    order: str | None = None
    layers: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    rules: list[_Rule] = Field(min_length=1)
    otherwise: Any = "self"
    colors: dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------ the result

@dataclass(frozen=True, eq=False)
class Automaton:
    name: str
    description: str
    dimensions: int                                  # 1 or 2
    states: StateSpace
    neighbour_offsets: tuple[tuple[int, ...], ...]
    edge_mode: str                                   # "wrap" or "fixed"
    edge_value: int | float | None                   # cell value beyond the grid ("fixed")
    parameters: dict[str, int | float]
    rules: tuple[ExpressionRule | TableRule | AgentRule, ...]
    otherwise: Expression                            # value when no rule applies
    colors: dict[int | float, tuple[int, int, int]]  # for pictures only
    radius: int                                      # how far the rules look
    source: Path                                     # the file it was read from
    digest: str                                      # SHA-256 of that file
    constants: dict[str, int | float] = field(default_factory=dict)  # state names + parameters
    update: str = "synchronous"                      # or "agents"
    order: str | None = None                         # agents: the order in which cells act
    layers: dict[str, StateSpace] = field(default_factory=dict)      # agents: extra numbers per cell
    layer_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)   # for pictures only

    @property
    def uses_random(self) -> bool:
        return (self.otherwise.uses_random or any(rule.uses_random for rule in self.rules)
                or self.order == "random")

    # A state is what one generation consists of. Without layers it is simply the grid.
    # With layers it is one array of whole numbers: state[0] is the grid, state[1:] the layers.
    @property
    def fields(self) -> tuple[str, ...]:
        """What every cell holds: "self" (its state) and one value per layer."""
        return ("self", *self.layers)

    @property
    def state_dtype(self):
        if not self.layers:
            return self.states.dtype
        spaces = (self.states, *self.layers.values())
        small = all(s.kind != "real" and s.lowest == 0 and s.highest is not None and s.highest <= 255 for s in spaces)
        return np.dtype(np.uint8) if small else self.states.work_dtype    # one byte per value when all fit

    def state_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (len(self.fields), *shape) if self.layers else shape

    def cells(self, state):
        """The grid of cell states inside a state (or inside a stack of states, one per step)."""
        return state[..., 0, *(slice(None),) * self.dimensions].astype(self.states.dtype) if self.layers else state

    def layer(self, state, name: str):
        return state[..., 1 + list(self.layers).index(name), *(slice(None),) * self.dimensions]

    @property
    def deterministic(self) -> bool:
        """True if the same grid always leads to the same next grid."""
        return not self.uses_random


def load_automaton(path: str | Path) -> Automaton:
    """Read, check and compile an automaton file."""
    path = Path(path)
    data = read_json(path)
    model = validate(_AutomatonFile, data, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return build_automaton(model, path, digest)


def build_automaton(model: _AutomatonFile, file: Path, digest: str) -> Automaton:
    def fail(where: str, message: str):
        raise InputFileError(file, where, message)

    if model.dimensions not in (1, 2):
        fail("dimensions", "must be 1 or 2")
    dims = model.dimensions

    if model.update not in UPDATE_MODES:
        fail("update", f"unknown update mode '{model.update}' (available: {', '.join(UPDATE_MODES)})")
    agents = model.update == "agents"
    order = model.order
    if agents:
        order = order or "random"
        if order not in ORDERS:
            fail("order", f"must be one of: {', '.join(ORDERS)}")
    else:
        for name in ("order", "layers"):
            if getattr(model, name):
                fail(name, "is only used by automata with \"update\": \"agents\"")

    # States -------------------------------------------------------------
    try:
        states = state_space_from_file(model.states)
    except ValueError as error:
        fail("states", str(error))
    for name in states.names:
        if name in RESERVED_NAMES or keyword.iskeyword(name):
            fail("states", f"'{name}' is a reserved word in rules; choose another name")

    # Neighbourhood and edges -------------------------------------------
    try:
        offsets = resolve_neighbourhood(model.neighbourhood, dims)
    except ValueError as error:
        fail("neighbourhood", str(error))

    edge_value = None
    if model.edges.type == "fixed":
        if model.edges.value is None:
            fail("edges.value", "fixed edges need a 'value' (the state beyond the grid)")
        try:
            edge_value = states.parse(model.edges.value)
        except ValueError as error:
            fail("edges.value", str(error))
    elif model.edges.type == "wrap":
        if model.edges.value is not None:
            fail("edges.value", "wrap-around edges take no value")
    else:
        fail("edges.type", "must be \"wrap\" or \"fixed\"")

    # Parameters ---------------------------------------------------------
    parameters: dict[str, int | float] = {}
    for name, value in model.parameters.items():
        where = f"parameters.{name}"
        if not name.isidentifier() or keyword.iskeyword(name):
            fail(where, "a parameter name must be a word made of letters, digits and _")
        if name in RESERVED_NAMES:
            fail(where, f"'{name}' is a reserved word in rules; choose another name")
        if name in states.names:
            fail(where, f"'{name}' is already a state name")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(where, "must be a number")
        parameters[name] = value

    # Layers (agents only) ------------------------------------------------
    layers: dict[str, StateSpace] = {}
    for name, raw in model.layers.items():
        where = f"layers.{name}"
        if not name.isidentifier() or keyword.iskeyword(name):
            fail(where, "a layer name must be a word made of letters, digits and _")
        if name in RESERVED_NAMES:
            fail(where, f"'{name}' is a reserved word in rules; choose another name")
        if name in states.names or name in parameters:
            fail(where, f"'{name}' is already a state or parameter name")
        try:
            layers[name] = layer_space_from_file(raw)
        except ValueError as error:
            fail(where, str(error))

    constants = {**states.constants(), **parameters}
    names = frozenset(constants) | {"self", "step"} | frozenset(DIRECTIONS[dims]) | frozenset(layers)
    state_names = frozenset(states.names)

    def compile_(text, where, other_cell: bool = False) -> Expression:
        return compile_expression(text, where, str(file), names, state_names, frozenset(layers), other_cell)

    # Rules --------------------------------------------------------------
    rules: list[ExpressionRule | TableRule | AgentRule] = []
    for index, rule in enumerate(model.rules):
        where = f"rules[{index}]"
        if agents:
            if rule.table is not None:
                fail(where, "tables are not available in agents mode; use 'when' with 'then' or 'do'")
            when = compile_(True if rule.when is None else rule.when, f"{where}.when")
            rules.append(build_agent_rule(when, rule.then, rule.do, where, str(file), ("self", *layers),
                                          offsets, dims, compile_))
        elif rule.do is not None:
            fail(f"{where}.do", "actions need \"update\": \"agents\" in the automaton")
        elif rule.table is not None:
            if rule.when is not None or rule.then is not None:
                fail(where, "a rule has either a 'table' or 'when'/'then', not both")
            rules.append(build_table_rule(rule.table, f"{where}.table", str(file), states, offsets))
        else:
            if rule.then is None:
                fail(where, "a rule needs 'then' (the new value) or a 'table'")
            when = compile_(True if rule.when is None else rule.when, f"{where}.when")
            then = compile_(rule.then, f"{where}.then")
            rules.append(ExpressionRule(where=where, when=when, then=then))
    otherwise = compile_(model.otherwise, "otherwise")
    if agents and model.otherwise != "self":
        fail("otherwise", "is not used in agents mode: a cell that no rule changes stays as it is")

    # Colours (pictures only): for states, and for layers (drawn where the layer is not 0)
    colors, layer_colors = {}, {}
    for key, colour in model.colors.items():
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
            fail(f"colors.{key}", "must be a colour like \"#ff8800\"")
        rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
        if key in layers:
            layer_colors[key] = rgb
            continue
        try:
            colors[states.parse(key)] = rgb
        except ValueError as error:
            fail(f"colors.{key}", str(error) + (f"; nor a layer ({', '.join(layers)})" if layers else ""))

    radius = max(1, max(abs(c) for offset in offsets for c in offset))
    return Automaton(
        name=model.name, description=model.description, dimensions=dims, states=states,
        neighbour_offsets=offsets, edge_mode=model.edges.type, edge_value=edge_value,
        parameters=parameters, rules=tuple(rules), otherwise=otherwise, colors=colors,
        radius=radius, source=file, digest=digest, constants=constants,
        update=model.update, order=order, layers=layers, layer_colors=layer_colors,
    )
