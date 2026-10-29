"""Grids written as text, used in experiment files and in printed answers.

A grid is a list of rows. A row is read in one of two ways:

    "TT.F"          no spaces: every character is one cell
    "0 12 3"        with spaces: every word is one cell (for long names or numbers)

Each cell is looked up in the legend first (e.g. {"X": "STATE_X"}) and otherwise read
directly as a state (a state name or a number). A 1D grid is exactly one row.
"""

import numpy as np

from .states import StateSpace


def split_row(text: str) -> list[str]:
    """'TT.F' -> ['T', 'T', '.', 'F'];  '0 12 3' -> ['0', '12', '3']"""
    text = text.strip()
    return text.split() if any(ch.isspace() for ch in text) else list(text)


def parse_legend(raw: dict, states: StateSpace) -> dict[str, int | float]:
    """{"X": "STATE_X"} -> {"X": 1}. Raises ValueError with a readable message."""
    legend = {}
    for token, value in raw.items():
        if not token or any(ch.isspace() for ch in token):
            raise ValueError(f"legend key {token!r} must be text without spaces")
        try:
            legend[token] = states.parse(value)
        except ValueError as error:
            raise ValueError(f"legend '{token}': {error}") from None
    return legend


def parse_grid(rows: list[str], states: StateSpace, dimensions: int,
               legend: dict[str, int | float] | None = None) -> np.ndarray:
    """Read a grid written as text rows. Raises ValueError with a readable message."""
    legend = legend or {}
    if dimensions == 1 and len(rows) != 1:
        raise ValueError(f"a 1D grid is written as exactly one row (got {len(rows)})")
    table = []
    for number, row in enumerate(rows, start=1):
        tokens = split_row(row)
        if not tokens:
            raise ValueError(f"row {number} is empty")
        values = []
        for token in tokens:
            if token in legend:
                values.append(legend[token])
                continue
            try:
                values.append(states.parse(token))
            except ValueError as error:
                raise ValueError(f"row {number}, '{token}': not in the legend, and {error}") from None
        table.append(values)
    if len({len(values) for values in table}) > 1:
        raise ValueError("all rows must have the same number of cells")
    grid = np.array(table, dtype=states.dtype)
    return grid[0] if dimensions == 1 else grid


def format_grid(grid: np.ndarray, states: StateSpace, legend: dict[str, int | float] | None = None) -> list[str]:
    """Write a grid as text rows, using the legend's symbols where possible."""
    symbol_for = {}
    for token, value in (legend or {}).items():
        symbol_for.setdefault(value, token)
    rows = grid[np.newaxis] if grid.ndim == 1 else grid
    tokens = [[symbol_for.get(value.item(), None) or states.format(value) for value in row] for row in rows]
    joiner = "" if all(len(token) == 1 for row in tokens for token in row) else " "
    return [joiner.join(row) for row in tokens]
