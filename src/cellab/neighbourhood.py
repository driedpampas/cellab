"""Which cells a cell can see, and what lies beyond the edges of the grid.

A neighbourhood is a list of offsets relative to a cell: (-1, 0) is the cell above,
(0, 1) the cell to the right, (-1,) the cell to the left in a 1D row.

To hand every cell its neighbours at once, the grid is padded with `radius` extra
cells on every side: copied from the opposite side for "wrap" edges, or filled with
a fixed value for "fixed" edges. Each neighbour is then a shifted view (a slice) of
the padded grid, so no loop over cells is ever needed.

    padded grid (radius 1, wrap)        the view for offset (-1, 0) ("north")
    . . . . . . .                       is the padded grid shifted down by one:
    . a b c d e .                       every cell sees the cell above it.
    . f g h i j .
    . . . . . . .
"""

import numpy as np

# name -> (dimensions, offsets in reading order)
NAMED_NEIGHBOURHOODS: dict[str, tuple[int, tuple[tuple[int, ...], ...]]] = {
    "elementary": (1, ((-1,), (1,))),
    "von_neumann": (2, ((-1, 0), (0, -1), (0, 1), (1, 0))),
    "moore": (2, tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))),
}

# Single neighbours that rules can name directly, whatever the neighbourhood is.
DIRECTIONS: dict[int, dict[str, tuple[int, ...]]] = {
    1: {"left": (-1,), "right": (1,)},
    2: {
        "north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1),
        "north_west": (-1, -1), "north_east": (-1, 1), "south_west": (1, -1), "south_east": (1, 1),
    },
}


def resolve_neighbourhood(raw, dimensions: int) -> tuple[tuple[int, ...], ...]:
    """Turn the `neighbourhood` field of a file into a tuple of offsets.

    Accepts a name ("elementary", "von_neumann", "moore") or an explicit list of
    offsets such as [[-1, 0], [1, 0]]. Raises ValueError with a readable message.
    """
    if isinstance(raw, str):
        if raw not in NAMED_NEIGHBOURHOODS:
            raise ValueError(f"unknown neighbourhood '{raw}' (known: {', '.join(NAMED_NEIGHBOURHOODS)}, "
                             "or a list of offsets)")
        dims, offsets = NAMED_NEIGHBOURHOODS[raw]
        if dims != dimensions:
            raise ValueError(f"'{raw}' is a {dims}D neighbourhood, but the automaton has {dimensions} dimension(s)")
        return offsets

    if isinstance(raw, list) and raw:
        offsets: list[tuple[int, ...]] = []
        for item in raw:
            if (not isinstance(item, list) or len(item) != dimensions
                    or not all(isinstance(v, int) and not isinstance(v, bool) for v in item)):
                raise ValueError(f"offset {item!r} must be a list of {dimensions} whole number(s)")
            offset = tuple(item)
            if not any(offset):
                raise ValueError("the cell itself (offset of zeros) is not a neighbour; use 'self' in rules")
            if offset in offsets:
                raise ValueError(f"offset {item!r} is listed twice")
            offsets.append(offset)
        return tuple(offsets)

    raise ValueError("must be a neighbourhood name or a non-empty list of offsets")


def pattern_offsets(offsets: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """The cells a table rule reads: the neighbours plus the cell itself, in reading order.

    For the elementary neighbourhood this is (left, self, right).
    """
    zero = tuple(0 for _ in offsets[0])
    return tuple(sorted(set(offsets) | {zero}))


class GridViews:
    """A padded copy of a grid that hands out the grid shifted by any offset."""

    def __init__(self, grid: np.ndarray, radius: int, edge_mode: str, edge_value):
        self.shape = grid.shape
        self.radius = radius
        self.edge_mode = edge_mode
        if edge_mode == "wrap":
            self.padded = np.pad(grid, radius, mode="wrap")
        else:
            self.padded = np.pad(grid, radius, mode="constant", constant_values=edge_value)

    def at(self, offset: tuple[int, ...]) -> np.ndarray:
        """The value of the neighbour at `offset`, for every cell at once."""
        slices = tuple(slice(self.radius + d, self.radius + d + n) for d, n in zip(offset, self.shape))
        return self.padded[slices]
