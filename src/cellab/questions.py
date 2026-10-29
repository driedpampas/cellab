"""The five questions from the project brief.

Each question is a small object with three jobs:

    capture_steps()   which grids it needs kept while the run goes on
    resolved(...)     whether the run has gathered enough to answer it
    answer(...)       the answer, as a sentence plus structured data

The run (runner.py) keeps going until every question is resolved, the experiment's
`steps` are done, or `max_steps` is reached. Answers come only from the fingerprint
history and the kept grids, never from the automaton's rules, so they work the
same for every automaton.

A deterministic run that repeats a grid is in a cycle forever after, so it tells
us the whole future: that is why a repeat resolves every question at once. Runs
with random rules give no such guarantee, and their answers say so.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fingerprints import History, fingerprint
from .grids import format_grid
from .states import StateSpace

RANDOM_NOTE = "the rules use random numbers, so this is not a guarantee about later steps"


@dataclass
class AnswerContext:
    """What a question may look at when it answers."""
    history: History
    captured: dict[int, np.ndarray]            # grids kept during the run, by step
    grid_at: Callable[[int], np.ndarray]       # any earlier grid (read back or recomputed)
    states: StateSpace
    legend: dict[str, int | float]
    deterministic: bool
    max_steps: int

    def grid_after(self, n: int) -> tuple[np.ndarray, int] | None:
        """The grid after n steps, and the step it was actually read from.

        Beyond the last simulated step this only works for a deterministic run
        that has entered a cycle: step n then shows the same grid as an earlier
        step inside the cycle.
        """
        if n <= self.history.last_step:
            grid = self.captured[n] if n in self.captured else self.grid_at(n)
            return grid, n
        if self.deterministic and self.history.cycle is not None:
            start, length = self.history.cycle
            same = start + (n - start) % length
            return self.grid_at(same), same
        return None

    def undecided(self, what: str) -> str:
        return (f"Not decided: {what} within {self.history.last_step} steps "
                f"(the run stops at max_steps = {self.max_steps}).")


def _steps(n: int) -> str:
    return "1 step" if n == 1 else f"{n} steps"


def _cells(count: int) -> str:
    return "is 1 cell" if count == 1 else f"are {count} cells"


@dataclass
class Answer:
    question: dict
    decided: bool
    text: str
    result: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"question": self.question, "decided": self.decided, "answer": self.text, "result": self.result}


class Question:
    ask = ""

    def to_dict(self) -> dict:
        return {"ask": self.ask}

    def capture_steps(self) -> set[int]:
        return set()

    def resolved(self, history: History, deterministic: bool) -> bool:
        raise NotImplementedError

    def answer(self, ctx: AnswerContext) -> Answer:
        raise NotImplementedError


@dataclass
class StateAfter(Question):
    """1. What is the state after N iterations?"""
    n: int
    ask = "state_after"

    def to_dict(self):
        return {"ask": self.ask, "n": self.n}

    def capture_steps(self):
        return {self.n}

    def resolved(self, history, deterministic):
        return history.last_step >= self.n or (deterministic and history.cycle is not None)

    def answer(self, ctx):
        found = ctx.grid_after(self.n)
        if found is None:
            return Answer(self.to_dict(), False, ctx.undecided(f"step {self.n} was not reached"))
        grid, source = found
        rows = format_grid(grid, ctx.states, ctx.legend)
        text = f"After {self.n} steps:\n" + "\n".join(rows)
        result = {"step": self.n, "grid": rows}
        if source != self.n:
            text += f"\n(the run is in a cycle, so step {self.n} shows the same grid as step {source})"
            result["same_as_step"] = source
        return Answer(self.to_dict(), True, text, result)


@dataclass
class CountAfter(Question):
    """4. How many cells of type X exist after N iterations?"""
    n: int
    value: int | float
    value_text: str
    ask = "count_after"

    def to_dict(self):
        return {"ask": self.ask, "n": self.n, "value": self.value_text}

    def capture_steps(self):
        return {self.n}

    def resolved(self, history, deterministic):
        return history.last_step >= self.n or (deterministic and history.cycle is not None)

    def answer(self, ctx):
        found = ctx.grid_after(self.n)
        if found is None:
            return Answer(self.to_dict(), False, ctx.undecided(f"step {self.n} was not reached"))
        grid, source = found
        count = int(np.count_nonzero(grid == self.value))
        text = f"After {_steps(self.n)} there {_cells(count)} with value {self.value_text}."
        result = {"step": self.n, "value": self.value_text, "count": count}
        if source != self.n:
            result["same_as_step"] = source
        return Answer(self.to_dict(), True, text, result)


@dataclass
class Stable(Question):
    """2. When does the system reach a stable state (no further changes)?"""
    ask = "stable"

    def resolved(self, history, deterministic):
        if deterministic:
            return history.cycle is not None
        return history.standstill is not None

    def answer(self, ctx):
        history = ctx.history
        if history.standstill is not None:
            start = history.standstill - 1
            if ctx.deterministic:
                text = f"Stable from step {start}: the grid never changes after that."
            else:
                text = f"The grid did not change from step {start} to {start + 1}; but {RANDOM_NOTE}."
            return Answer(self.to_dict(), True, text, {"stable": True, "from_step": start})
        if ctx.deterministic and history.cycle is not None:
            start, length = history.cycle
            text = (f"Never stable: from step {start} the grids repeat in a cycle of length {length}, "
                    "so the grid keeps changing forever.")
            return Answer(self.to_dict(), True, text,
                          {"stable": False, "cycle_start": start, "cycle_length": length})
        return Answer(self.to_dict(), False, ctx.undecided("no stable state was found"), {"stable": None})


@dataclass
class Cycle(Question):
    """5. Does the simulation enter a cycle (a repeating sequence of states)?"""
    ask = "cycle"

    def resolved(self, history, deterministic):
        return history.repeat is not None

    def answer(self, ctx):
        history = ctx.history
        if history.repeat is None:
            return Answer(self.to_dict(), False, ctx.undecided("no grid repeated"), {"cycle": None})
        earlier, later = history.repeat
        length = later - earlier
        result = {"cycle": True, "start": earlier, "length": length}
        if ctx.deterministic:
            text = (f"Yes: from step {earlier} the grids repeat every {length} step(s) "
                    f"(step {later} is the same grid as step {earlier}).")
            if length == 1:
                text += " A cycle of length 1 means the grid is stable."
        else:
            text = f"The grid at step {later} is the same as at step {earlier}; but {RANDOM_NOTE}."
            result["guaranteed"] = False
        return Answer(self.to_dict(), True, text, result)


@dataclass
class Occurs(Question):
    """3. Does a particular configuration ever occur?"""
    grid: np.ndarray
    rows: list[str]
    ask = "occurs"
    fingerprint: bytes = b""

    def __post_init__(self):
        self.fingerprint = fingerprint(self.grid)

    def to_dict(self):
        return {"ask": self.ask, "rows": self.rows}

    def resolved(self, history, deterministic):
        return self.fingerprint in history.first_seen or (deterministic and history.cycle is not None)

    def answer(self, ctx):
        history = ctx.history
        if self.fingerprint in history.first_seen:
            step = history.first_seen[self.fingerprint]
            return Answer(self.to_dict(), True, f"Yes, first at step {step}.", {"occurs": True, "first_step": step})
        if ctx.deterministic and history.cycle is not None:
            _, later = history.repeat
            text = (f"No, never: by step {later} the run had entered a cycle, so every grid it will "
                    "ever show had already appeared, and this configuration was not one of them.")
            return Answer(self.to_dict(), True, text, {"occurs": False, "certain": True})
        text = ctx.undecided("the configuration did not appear")
        if not ctx.deterministic:
            text += f" Note: {RANDOM_NOTE}."
        return Answer(self.to_dict(), False, text, {"occurs": None})
