"""The two kinds of rules. Both answer the same question for every cell at once:

    "does this rule apply here, and if so, what is the new value?"

and return it as two arrays the size of the grid: `applies` (True/False) and
`values`. The engine (engine.py) combines the rules: the first rule that applies
to a cell decides its new value.

ExpressionRule   {"when": "<condition>", "then": "<new value>"}
TableRule        {"table": {"111": 0, "110": 0, ...}}
                 The table lists patterns of (neighbours + the cell itself, in
                 reading order) and the value each pattern produces. Patterns not
                 listed fall through to the next rule.
"""

from dataclasses import dataclass

import numpy as np

from .errors import InputFileError
from .expressions import EvalContext, Expression
from .grids import split_row
from .neighbourhood import pattern_offsets
from .states import StateSpace

MAX_TABLE_SIZE = 1_000_000   # number of possible patterns a table may cover


@dataclass(frozen=True, eq=False)
class ExpressionRule:
    where: str
    when: Expression
    then: Expression

    @property
    def uses_random(self) -> bool:
        return self.when.uses_random or self.then.uses_random

    def evaluate(self, ctx: EvalContext) -> tuple[np.ndarray, np.ndarray]:
        # Both parts are always evaluated, so random() draws happen in a fixed order.
        applies = np.broadcast_to(np.asarray(self.when.evaluate(ctx), dtype=bool), ctx.shape)
        values = np.broadcast_to(self.then.evaluate(ctx), ctx.shape)
        return applies, values


@dataclass(frozen=True, eq=False)
class TableRule:
    where: str
    offsets: tuple[tuple[int, ...], ...]   # cells read, in pattern order
    base: int                              # number of possible values per cell
    lowest: int                            # the smallest possible value
    listed: np.ndarray                     # listed[code]  -> is this pattern in the table?
    results: np.ndarray                    # results[code] -> the value it produces

    uses_random = False

    def evaluate(self, ctx: EvalContext) -> tuple[np.ndarray, np.ndarray]:
        # Each pattern becomes a number ("code"), like reading the cells as digits in base `base`.
        code = np.zeros(ctx.shape, dtype=np.int64)
        for offset in self.offsets:
            cell = ctx.grid if not any(offset) else ctx.views.at(offset)
            code = code * self.base + (cell.astype(np.int64) - self.lowest)
        return self.listed[code], self.results[code]


def build_table_rule(table: dict, where: str, file: str, states: StateSpace,
                     neighbour_offsets: tuple[tuple[int, ...], ...]) -> TableRule:
    """Check a table from a file and turn it into lookup arrays."""
    def fail(message: str):
        raise InputFileError(file, where, message)

    if states.kind == "real":
        fail("tables need named states or integer states with a min and a max")
    if states.kind == "integer" and (states.minimum is None or states.maximum is None):
        fail("tables on integer states need both 'min' and 'max' in \"states\"")
    if not table:
        fail("the table is empty")

    offsets = pattern_offsets(neighbour_offsets)
    lowest = int(states.lowest)
    base = int(states.highest) - lowest + 1
    size = base ** len(offsets)
    if size > MAX_TABLE_SIZE:
        fail(f"the table would cover {size:,} possible patterns; use when/then rules instead")

    listed = np.zeros(size, dtype=bool)
    results = np.zeros(size, dtype=np.int64)
    for pattern, result in table.items():
        tokens = split_row(pattern)
        if len(tokens) != len(offsets):
            fail(f"pattern '{pattern}' has {len(tokens)} cells, but each pattern lists "
                 f"{len(offsets)} (the neighbours and the cell itself, in reading order)")
        code = 0
        for token in tokens:
            try:
                value = int(states.parse(token))
            except ValueError as error:
                fail(f"pattern '{pattern}': {error}")
            code = code * base + (value - lowest)
        if listed[code]:
            fail(f"pattern '{pattern}' is listed twice")
        try:
            results[code] = int(states.parse(result))
        except ValueError as error:
            fail(f"pattern '{pattern}' -> {result!r}: {error}")
        listed[code] = True
    return TableRule(where=where, offsets=offsets, base=base, lowest=lowest, listed=listed, results=results)
