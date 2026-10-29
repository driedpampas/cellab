"""One step of an automaton: old grid in, new grid out.

This is the whole simulation. Everything is computed from the OLD grid, for all
cells at the same time, so the order of cells never matters.

    1. Widen the grid to a roomy number type (no overflow surprises in rules).
    2. Pad it and prepare the neighbour views                    (neighbourhood.py)
    3. Start every cell from the `otherwise` value (usually: unchanged).
    4. Go through the rules in file order. A rule only changes cells that no
       earlier rule has claimed: the first rule that applies to a cell wins.
    5. Check the new values (whole numbers, inside the declared states) and
       store them in the grid's normal type.                      (states.py)

The function never touches files, and its only side effect is drawing random
numbers from the generator it is given. That is what makes runs reproducible.
"""

import numpy as np

from .automaton import Automaton
from .errors import RuleError
from .expressions import EvalContext
from .neighbourhood import GridViews


def step(automaton: Automaton, grid: np.ndarray, rng: np.random.Generator, step_number: int) -> np.ndarray:
    """Compute the generation after `grid`. `step_number` is the number of `grid` itself."""
    states = automaton.states
    values = grid.astype(states.work_dtype)
    views = GridViews(values, automaton.radius, automaton.edge_mode, automaton.edge_value)
    ctx = EvalContext(values, views, automaton.neighbour_offsets, automaton.dimensions,
                      automaton.constants, states.constants(), rng, step_number)

    work = np.empty(grid.shape, dtype=states.work_dtype)
    work[...] = _cell_values(np.broadcast_to(automaton.otherwise.evaluate(ctx), grid.shape),
                             states.kind, automaton.otherwise.where)

    decided = np.zeros(grid.shape, dtype=bool)
    for rule in automaton.rules:
        applies, new_values = rule.evaluate(ctx)
        chosen = applies & ~decided            # cells this rule decides
        if chosen.any():
            work[chosen] = _cell_values(new_values[chosen], states.kind, rule.where)
            decided |= chosen

    try:
        return states.finish_step(work)
    except RuleError as error:
        raise RuleError(f"step {step_number} -> {step_number + 1}: {error}") from None


def _cell_values(values: np.ndarray, kind: str, where: str) -> np.ndarray:
    """Check values a rule produced before they are stored in a grid."""
    values = np.asarray(values)
    if kind == "real":
        if np.isnan(values).any():
            raise RuleError(f"{where}: produced a value that is not a number (NaN)")
        return values
    if values.dtype.kind == "f":
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            bad = values[~(np.isfinite(values) & (values == np.floor(values)))][0]
            raise RuleError(f"{where}: produced {bad:g}, which is not a whole number "
                            "(use // instead of /, or round())")
        return values.astype(np.int64)
    return values
