"""The values a cell can take, and how those values are written in files.

There are three kinds of state space:

- named:   a list of names, e.g. ["STATE_A", "STATE_B"]; stored as small integers 0, 1, ...
- integer: whole numbers, optionally limited by a minimum and a maximum
- real:    decimal numbers, optionally rounded to some decimals after every step

Inside the engine every grid is a plain NumPy array of numbers. Names only exist
at the edges of the system: when files are read and when results are printed.
"""

from dataclasses import dataclass

import numpy as np

from .errors import RuleError


@dataclass(frozen=True)
class StateSpace:
    kind: str                      # "named", "integer" or "real"
    names: tuple[str, ...] = ()    # only for "named"
    minimum: float | None = None   # optional limits for "integer" and "real"
    maximum: float | None = None
    decimals: int | None = None    # optional rounding for "real"

    # ----------------------------------------------------------------- storage
    @property
    def dtype(self) -> np.dtype:
        """The NumPy type used to store grids of this state space."""
        if self.kind == "named":
            return np.dtype(np.uint8) if len(self.names) <= 256 else np.dtype(np.uint16)
        if self.kind == "integer":
            return np.dtype(np.int64)
        return np.dtype(np.float64)

    @property
    def work_dtype(self) -> np.dtype:
        """The type used while a step is computed (wide enough to detect bad values)."""
        return np.dtype(np.float64) if self.kind == "real" else np.dtype(np.int64)

    @property
    def lowest(self) -> float | None:
        return 0 if self.kind == "named" else self.minimum

    @property
    def highest(self) -> float | None:
        return len(self.names) - 1 if self.kind == "named" else self.maximum

    def constants(self) -> dict[str, int]:
        """State names usable in rules, mapped to the number that stores them."""
        return {name: index for index, name in enumerate(self.names)}

    # ------------------------------------------------------- values from files
    def parse(self, value) -> int | float:
        """Turn a value written in a file (a state name or a number) into a cell value.

        Raises ValueError with a readable message; callers add the file and field.
        """
        if self.kind == "named":
            if isinstance(value, str) and value in self.names:
                return self.names.index(value)
            raise ValueError(f"'{value}' is not a state (states are {', '.join(self.names)})")

        number = value
        if isinstance(value, str):
            try:
                number = float(value) if self.kind == "real" else int(value)
            except ValueError:
                raise ValueError(f"'{value}' is not a number") from None
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"'{value}' is not a number")
        if self.kind == "integer":
            if number != int(number):
                raise ValueError(f"{value} is not a whole number")
            number = int(number)
        else:
            number = float(number)
        self._check_limits(number)
        return number

    def format(self, value) -> str:
        """Write a cell value the way a person would read it."""
        if self.kind == "named":
            return self.names[int(value)]
        if self.kind == "integer":
            return str(int(value))
        return f"{float(value):g}"

    def _check_limits(self, number: float) -> None:
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"{number} is below the minimum {self.minimum}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{number} is above the maximum {self.maximum}")

    # ------------------------------------------------------- values from rules
    def finish_step(self, work: np.ndarray) -> np.ndarray:
        """Check the values a step produced and store them in the final grid type."""
        if self.kind == "real" and self.decimals is not None:
            work = np.round(work, self.decimals)
        low, high = self.lowest, self.highest
        if low is not None and (work < low).any():
            raise RuleError(f"a rule produced the value {_plain(work.min())}, "
                            f"below the lowest allowed value {self.format(low)}")
        if high is not None and (work > high).any():
            raise RuleError(f"a rule produced the value {_plain(work.max())}, "
                            f"above the highest allowed value {self.format(high)}")
        return work.astype(self.dtype)


def checked_values(values, kind: str, where: str) -> np.ndarray:
    """Check values a rule produced before they are stored: numbers only, whole unless `kind` is real."""
    values = np.asarray(values)
    if kind == "real":
        if np.isnan(values).any():
            raise RuleError(f"{where}: produced a value that is not a number (NaN)")
        return values
    if values.dtype.kind == "f":
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            bad = values[~(np.isfinite(values) & (values == np.floor(values)))].flat[0]
            raise RuleError(f"{where}: produced {bad:g}, which is not a whole number "
                            "(use // instead of /, or round())")
        return values.astype(np.int64)
    return values


def _plain(number) -> str:
    """Show a number without NumPy decoration (7 rather than np.int64(7))."""
    number = float(number)
    return str(int(number)) if number == int(number) else f"{number:g}"


def state_space_from_file(raw) -> StateSpace:
    """Build a StateSpace from the `states` field of an automaton file.

    Accepted forms:
        ["STATE_A", "STATE_B"]
        {"type": "integer", "min": 0, "max": 3}
        {"type": "real", "min": 0, "decimals": 2}
    Raises ValueError with a readable message.
    """
    if isinstance(raw, list):
        if not raw:
            raise ValueError("the list of states is empty")
        for name in raw:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"state name {name!r} must be a word made of letters, digits and _ (not starting with a digit)")
        if len(set(raw)) != len(raw):
            raise ValueError("state names must be unique")
        if len(raw) > 65535:
            raise ValueError("too many states")
        return StateSpace(kind="named", names=tuple(raw))

    if isinstance(raw, dict):
        allowed = {"type", "min", "max", "decimals"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown field(s) {', '.join(sorted(unknown))}; allowed: {', '.join(sorted(allowed))}")
        kind = raw.get("type")
        if kind not in ("integer", "real"):
            raise ValueError("'type' must be \"integer\" or \"real\" (or give a list of state names)")
        low, high = raw.get("min"), raw.get("max")
        for label, limit in (("min", low), ("max", high)):
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))):
                raise ValueError(f"'{label}' must be a number")
            if kind == "integer" and limit is not None and limit != int(limit):
                raise ValueError(f"'{label}' must be a whole number for integer states")
        if low is not None and high is not None and low > high:
            raise ValueError("'min' is larger than 'max'")
        decimals = raw.get("decimals")
        if decimals is not None:
            if kind != "real":
                raise ValueError("'decimals' is only allowed for real states")
            if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
                raise ValueError("'decimals' must be a whole number, 0 or more")
        return StateSpace(kind=kind, minimum=low, maximum=high, decimals=decimals)

    raise ValueError("must be a list of state names or an object like {\"type\": \"integer\"}")


def layer_space_from_file(raw) -> StateSpace:
    """Build the StateSpace of a layer: whole numbers from `min` (default 0) to an optional `max`.

    Accepted form: {} or {"min": 0, "max": 20}. Raises ValueError with a readable message.
    """
    if not isinstance(raw, dict):
        raise ValueError("must be an object like {\"max\": 20} (or {} for any whole number from 0)")
    unknown = set(raw) - {"min", "max"}
    if unknown:
        raise ValueError(f"unknown field(s) {', '.join(sorted(unknown))}; allowed: max, min")
    space = state_space_from_file({"type": "integer", "min": raw.get("min", 0), "max": raw.get("max")})
    if space.minimum > 0 or (space.maximum is not None and space.maximum < 0):
        raise ValueError("the range must include 0 (the value of a layer where nothing is stored)")
    return space
