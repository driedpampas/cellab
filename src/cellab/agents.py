"""The "agents" update mode: for automata where things move, eat, multiply or die.

In the synchronous mode (engine.py) every cell only decides its own new value.
That cannot express "this thing goes over there": two cells change because of one
decision, and two things may want the same place. The agents mode adds exactly
that, and nothing else:

    {"when": "<condition>",
     "do": [ {"to":    {"among": [...], "where": "<condition>", "choose": "random"},
              "here":  {"self": "<value>", "<layer>": "<value>"},
              "there": {"self": "<value>", "<layer>": "<value>"}},
             ... further actions, tried only if the ones before could not happen ... ]}

One step has two parts:

1. DECIDE, for all cells at once, from the old grid (the same machinery as the
   synchronous mode): the first rule whose `when` holds for a cell is that cell's
   rule for this step.
2. ACT, one cell at a time, on the live grid. A cell tries its rule's actions in
   order and performs the first one that is possible. An action with `to` looks
   for another cell (among the listed neighbours, or anywhere on the grid) that
   satisfies `where`, and is impossible if there is none. `here` gives new
   values to the acting cell and `there` to the chosen cell; all of them are
   computed from the grid as it is just before the action.

A cell that another cell's action has written to loses its turn in that step:
whatever arrives or is born there acts from the next step on, and whatever was
there is gone.

Cells hold one state as always. Automata that need more per cell (energy, an
age, a heading) declare "layers": extra whole numbers stored next to the state.
The whole world is then one array, values[field][position], with field 0 the
state and one more field per layer.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .errors import InputFileError, RuleError
from .expressions import EvalContext, Expression
from .loading import validate
from .neighbourhood import DIRECTIONS, GridViews
from .states import checked_values

ORDERS = ("random", "reading", "reverse")      # the order in which cells act within a step
CHOICES = ("random", "first")                  # which suitable cell an action takes


# ------------------------------------------------------------ file structure

class _Choice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    among: Any = "neighbours"
    where: Any = True
    choose: str = "random"


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: _Choice | None = None
    here: dict[str, Any] = Field(default_factory=dict)
    there: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------ compiled rules

@dataclass(frozen=True, eq=False)
class Choice:
    """How an action finds its other cell."""
    offsets: np.ndarray | None      # one row per candidate, relative to the acting cell; None = anywhere
    where: Expression               # which candidates are suitable
    first: bool                     # take the first suitable one (otherwise a random one)


@dataclass(frozen=True, eq=False)
class Action:
    to: Choice | None
    here: tuple[tuple[int, Expression], ...]     # (field number, new value) for the acting cell
    there: tuple[tuple[int, Expression], ...]    # the same for the chosen cell

    @property
    def uses_random(self) -> bool:
        expressions = [e for _, e in self.here + self.there] + ([self.to.where] if self.to else [])
        return any(e.uses_random for e in expressions) or (self.to is not None and not self.to.first)


@dataclass(frozen=True, eq=False)
class AgentRule:
    where: str
    when: Expression
    actions: tuple[Action, ...]

    @property
    def uses_random(self) -> bool:
        return self.when.uses_random or any(action.uses_random for action in self.actions)


def build_agent_rule(when: Expression, then, do, where: str, file: str, fields: tuple[str, ...],
                     neighbour_offsets, dimensions: int, compile_: Callable) -> AgentRule:
    """Check one rule of an agents automaton. `fields` are "self" and the layer names;
    `compile_(text, where, other_cell)` compiles an expression of this automaton."""
    def fail(place: str, message: str):
        raise InputFileError(file, place, message)

    if then is not None:                         # {"when": ..., "then": X} is short for one action
        if do is not None:
            fail(where, "a rule has either 'then' or 'do', not both")
        do = {"here": {"self": then}}
    if do is None:
        fail(where, "a rule needs 'then' (a new value for the cell) or 'do' (actions)")
    raw_actions = do if isinstance(do, list) else [do]
    if not raw_actions:
        fail(f"{where}.do", "lists no actions")

    def assignments(raw: dict, place: str, other_cell: bool):
        for name in raw:
            if name not in fields:
                fail(f"{place}.{name}", f"is not something a cell holds (use: {', '.join(fields)})")
        return tuple((fields.index(name), compile_(text, f"{place}.{name}", other_cell)) for name, text in raw.items())

    actions = []
    for number, raw in enumerate(raw_actions):
        place = f"{where}.do[{number}]" if isinstance(do, list) else f"{where}.do"
        model = validate(_Action, raw, file, prefix=place)
        if not model.here and not model.there:
            fail(place, "an action needs 'here' and/or 'there' (the new values)")
        if model.there and model.to is None:
            fail(f"{place}.there", "needs 'to' (how the other cell is chosen)")
        choice = None
        if model.to is not None:
            if model.to.choose not in CHOICES:
                fail(f"{place}.to.choose", f"must be one of: {', '.join(CHOICES)}")
            try:
                offsets = _candidate_offsets(model.to.among, neighbour_offsets, dimensions)
            except ValueError as error:
                fail(f"{place}.to.among", str(error))
            choice = Choice(offsets=offsets, where=compile_(model.to.where, f"{place}.to.where", True),
                            first=model.to.choose == "first")
        other_cell = choice is not None
        actions.append(Action(to=choice, here=assignments(model.here, f"{place}.here", other_cell),
                              there=assignments(model.there, f"{place}.there", other_cell)))
    return AgentRule(where=where, when=when, actions=tuple(actions))


def _candidate_offsets(among, neighbour_offsets, dimensions: int) -> np.ndarray | None:
    """"neighbours", "anywhere", or a list of direction names / offsets -> array of offsets (None = anywhere)."""
    if among == "neighbours":
        return np.array(neighbour_offsets)
    if among == "anywhere":
        return None
    directions = DIRECTIONS[dimensions]
    if not isinstance(among, list) or not among:
        raise ValueError("must be \"neighbours\", \"anywhere\", or a list of directions such as "
                         f"[{', '.join(repr(name) for name in list(directions)[:2])}] or offsets")
    offsets = []
    for item in among:
        if isinstance(item, str) and item in directions:
            offsets.append(directions[item])
        elif (isinstance(item, list) and len(item) == dimensions and any(item)
              and all(isinstance(v, int) and not isinstance(v, bool) for v in item)):
            offsets.append(tuple(item))
        else:
            raise ValueError(f"{item!r} is not a direction ({', '.join(directions)}) "
                             f"or an offset of {dimensions} whole number(s)")
    return np.array(offsets)


# ------------------------------------------------------------ one step

def step(automaton, state: np.ndarray, rng: np.random.Generator, step_number: int) -> np.ndarray:
    """Compute the generation after `state` (see the two parts in the module text)."""
    layered = bool(automaton.layers)
    old = (state if layered else state[np.newaxis]).astype(automaton.states.work_dtype)
    shape = old.shape[1:]

    # 1. Decide: which rule each cell follows in this step, from the old grid.
    views = GridViews(old[0], automaton.radius, automaton.edge_mode, automaton.edge_value)
    ctx = EvalContext(old[0], views, automaton.neighbour_offsets, automaton.dimensions, automaton.constants,
                      automaton.states.constants(), rng, step_number, layers=dict(zip(automaton.layers, old[1:])))
    rule_of = np.full(shape, -1)
    for number, rule in enumerate(automaton.rules):
        applies = np.broadcast_to(np.asarray(rule.when.evaluate(ctx), dtype=bool), shape)
        rule_of[applies & (rule_of < 0)] = number          # the first rule that applies wins
    acting = np.argwhere(rule_of >= 0)                     # in reading order
    if automaton.order == "reverse":
        acting = acting[::-1]
    elif automaton.order == "random":
        acting = rng.permutation(acting)

    # 2. Act: one cell at a time, on the live grid.
    world = _World(automaton, old.copy(), rng, step_number)
    written = np.zeros(shape, dtype=bool)                  # cells another cell's action has written to
    for position in map(tuple, acting.tolist()):
        if written[position]:
            continue
        for action in automaton.rules[rule_of[position]].actions:
            happened, other = world.perform(action, position)
            if happened:
                if other is not None:
                    written[other] = True
                break

    new = world.values.astype(automaton.state_dtype)
    return new if layered else new[0]


class _World:
    """The grid while the cells act: values[field][position], read and changed cell by cell."""

    def __init__(self, automaton, values: np.ndarray, rng: np.random.Generator, step_number: int):
        self.automaton, self.values, self.rng, self.step = automaton, values, rng, step_number
        self.shape = values.shape[1:]
        self.fields = automaton.fields
        self.spaces = (automaton.states, *automaton.layers.values())
        self.wraps = automaton.edge_mode == "wrap"
        self.beyond = dict(zip(self.fields, (automaton.edge_value, *(0 for _ in automaton.layers))))
        self._everywhere: np.ndarray | None = None

    def cell(self, position: tuple[int, ...]) -> dict:
        """Everything the cell at `position` holds, by field name."""
        return dict(zip(self.fields, self.values[(slice(None), *position)]))

    def shifted(self, position: tuple[int, ...], offset) -> tuple[int, ...] | None:
        """The position `offset` away, or None if that is beyond a fixed edge."""
        target = tuple(p + d for p, d in zip(position, offset))
        if self.wraps:
            return tuple(t % n for t, n in zip(target, self.shape))
        return target if all(0 <= t < n for t, n in zip(target, self.shape)) else None

    def candidates(self, choice: Choice, position: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """The cells an action may choose from, and each one's number in the list ("direction")."""
        if choice.offsets is None:
            if self._everywhere is None:
                self._everywhere = np.argwhere(np.ones(self.shape, dtype=bool))
            cells = self._everywhere[(self._everywhere != position).any(axis=1)]
            return cells, np.arange(len(cells))
        cells = np.array(position) + choice.offsets
        directions = np.arange(len(cells))
        if self.wraps:
            return cells % self.shape, directions
        inside = ((cells >= 0) & (cells < self.shape)).all(axis=1)
        return cells[inside], directions[inside]

    def perform(self, action: Action, position: tuple[int, ...]) -> tuple[bool, tuple[int, ...] | None]:
        """Carry out `action` for the cell at `position` if possible.
        Returns (did it happen, the other cell it wrote to or None)."""
        ctx = _CellContext(self, position)
        other, nb = None, None
        if action.to is not None:
            cells, directions = self.candidates(action.to, position)
            if not len(cells):
                return False, None
            ctx.direction = directions
            candidates = dict(zip(self.fields, self.values[(slice(None), *cells.T)]))
            suitable = np.flatnonzero(np.broadcast_to(
                np.asarray(action.to.where.evaluate(ctx, candidates), dtype=bool), len(cells)))
            if not suitable.size:
                return False, None
            chosen = suitable[0] if action.to.first else suitable[self.rng.integers(suitable.size)]
            other = tuple(cells[chosen].tolist())
            ctx.direction, nb = directions[chosen], self.cell(other)

        # Every new value is computed from the grid as it is now; only then is anything written.
        writes = [(place, field, self._new_value(field, expression, ctx, nb))
                  for place, assignments in ((position, action.here), (other, action.there))
                  for field, expression in assignments]
        for place, field, value in writes:
            self.values[field][place] = value
        return True, other if action.there else None

    def _new_value(self, field: int, expression: Expression, ctx: "_CellContext", nb: dict | None):
        label = f"{expression.where} ('{expression.text}')"
        value = checked_values(expression.evaluate(ctx, nb), self.spaces[field].kind, label)
        if value.shape != ():
            raise RuleError(f"{label}: gives several values where one is needed")
        try:
            return self.spaces[field].finish_step(value)
        except RuleError as error:
            raise RuleError(f"{label}: {error}") from None


class _CellContext:
    """What an expression reads while one cell acts: that cell's live values.
    Offers the same interface as EvalContext, for a single cell instead of the whole grid."""

    shape = ()

    def __init__(self, world: _World, position: tuple[int, ...]):
        self.world, self.position = world, position
        self.rng = world.rng
        self.state_values = world.automaton.states.constants()
        self.direction = None                      # set while and after another cell is chosen

    def value_of(self, name: str):
        world = self.world
        if name in world.fields:
            return world.values[world.fields.index(name)][self.position]
        if name == "step":
            return world.step
        if name == "direction":
            return self.direction
        directions = DIRECTIONS[world.automaton.dimensions]
        if name in directions:
            return self._at(directions[name])["self"]
        return world.automaton.constants[name]

    def neighbours(self) -> list[dict]:
        return [self._at(offset) for offset in self.world.automaton.neighbour_offsets]

    def _at(self, offset) -> dict:
        position = self.world.shifted(self.position, offset)
        return self.world.beyond if position is None else self.world.cell(position)
