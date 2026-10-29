"""The rule language: small expressions evaluated for the whole grid at once.

A rule such as   self == STATE_A and count(STATE_B) >= 2   is written in ordinary
Python syntax. Python's own parser (the `ast` module) turns the text into a tree,
`compile_expression` checks that the tree only uses a small, safe set of
constructs, and `Expression.evaluate` walks the tree with NumPy arrays. Every name
stands for a whole grid of values, so one evaluation computes the result for every
cell at the same time.

Names an expression can use
    self                    the cell's current value
    left, right             (1D) the neighbour on that side
    north, south, east, west, north_east, north_west, south_east, south_west   (2D)
    step                    the number of the generation being read (0, 1, 2, ...)
    <state names>           as declared in "states" (named states only)
    <parameter names>       as declared in "parameters"
    nb                      only inside an aggregate: "each neighbour in turn"

Functions
    count(x)                how many neighbours make x true; count(NAME) means count(nb == NAME)
    sum(x), mean(x), min(x), max(x)    of x over the neighbours (x normally uses nb)
    min(a, b, ...), max(a, b, ...)     with two or more arguments: cell by cell
    random()                a number in [0, 1) from the run's seeded generator
    abs(x), floor(x), round(x), round(x, digits)

Operators: + - * / // % **, comparisons (chains allowed), and, or, not,
x if condition else y, and the bit operators & | ^ ~ << >>.

`and`/`or` always evaluate every part (no short-circuit). This keeps the number
of random() draws per step fixed, which is what makes a seed reproducible.
"""

import ast
import difflib
import functools
from dataclasses import dataclass

import numpy as np

from .errors import InputFileError, RuleError
from .neighbourhood import DIRECTIONS, GridViews

AGGREGATES = ("count", "sum", "mean", "min", "max")
FUNCTIONS = ("random", "abs", "floor", "round", "min", "max")
DIRECTION_NAMES = tuple(name for names in DIRECTIONS.values() for name in names)
RESERVED_NAMES = frozenset({"self", "step", "nb", "True", "False", *AGGREGATES, *FUNCTIONS, *DIRECTION_NAMES})

_BINARY = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide,
    ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power,
    ast.BitAnd: np.bitwise_and, ast.BitOr: np.bitwise_or, ast.BitXor: np.bitwise_xor,
    ast.LShift: np.left_shift, ast.RShift: np.right_shift,
}
_COMPARE = {
    ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less,
    ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
}
_UNARY = (ast.Not, ast.USub, ast.UAdd, ast.Invert)
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Name, ast.Constant, ast.Load, ast.And, ast.Or,
    *_BINARY, *_COMPARE, *_UNARY,
)


# ============================================================== compiling

@dataclass(frozen=True, eq=False)
class Expression:
    """A checked expression, ready to be evaluated on any grid."""

    text: str           # the expression as written in the file
    where: str          # the field it came from, e.g. "rules[1].when"
    tree: ast.expr      # the parsed and checked syntax tree
    uses_random: bool   # True if it calls random()

    def evaluate(self, ctx: "EvalContext"):
        """Evaluate for every cell at once. Returns an array (or a single number)."""
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                return _evaluate(self.tree, ctx, None)
        except (ArithmeticError, TypeError, ValueError) as error:
            raise RuleError(f"{self.where} ('{self.text}'): {error}") from None


def compile_expression(text, where: str, file: str, names: frozenset[str],
                       state_names: frozenset[str] = frozenset()) -> Expression:
    """Parse and check one expression from a file.

    `names` are all the names the expression may use; `state_names` are the
    declared state names (needed for the count(NAME) shorthand).
    Raises InputFileError naming the file and the field.
    """
    if isinstance(text, bool):
        text = str(text)
    elif isinstance(text, (int, float)):
        text = repr(text)
    if not isinstance(text, str) or not text.strip():
        raise InputFileError(file, where, "must be a non-empty expression")

    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as error:
        raise InputFileError(file, where, f"'{text}' is not a valid expression ({error.msg})") from None

    checker = _Checker(text=text, where=where, file=file, names=names, state_names=state_names)
    checker.check(tree, inside_aggregate=False)
    return Expression(text=text, where=where, tree=tree, uses_random=checker.uses_random)


class _Checker:
    """Walks a parsed expression and rejects anything outside the rule language."""

    def __init__(self, text, where, file, names, state_names):
        self.text, self.where, self.file = text, where, file
        self.names, self.state_names = names, state_names
        self.uses_random = False

    def fail(self, message: str):
        raise InputFileError(self.file, self.where, f"'{self.text}': {message}")

    def check(self, node: ast.AST, inside_aggregate: bool) -> None:
        if isinstance(node, ast.Call):
            self.check_call(node, inside_aggregate)
            return
        if not isinstance(node, _ALLOWED_NODES):
            self.fail(f"{type(node).__name__} is not allowed in rules")
        if isinstance(node, ast.Constant) and (isinstance(node.value, str) or not isinstance(node.value, (int, float))):
            self.fail("only numbers and True/False can be written literally (write state names without quotes)")
        if isinstance(node, ast.Name):
            self.check_name(node.id, inside_aggregate)
        for child in ast.iter_child_nodes(node):
            self.check(child, inside_aggregate)

    def check_name(self, name: str, inside_aggregate: bool) -> None:
        if name == "nb":
            if not inside_aggregate:
                self.fail("'nb' can only be used inside count(), sum(), mean(), min() or max()")
            return
        if name not in self.names:
            by_lower = {known.lower(): known for known in sorted(self.names)}
            close = difflib.get_close_matches(name.lower(), list(by_lower), n=1)
            hint = f" (did you mean '{by_lower[close[0]]}'?)" if close else ""
            self.fail(f"unknown name '{name}'{hint}")

    def check_call(self, node: ast.Call, inside_aggregate: bool) -> None:
        if not isinstance(node.func, ast.Name):
            self.fail("only the built-in functions can be called")
        name, args = node.func.id, node.args
        if node.keywords:
            self.fail(f"{name}() does not take named arguments")

        if name in AGGREGATES and len(args) == 1:
            if inside_aggregate:
                self.fail(f"{name}() cannot be used inside another count/sum/mean/min/max")
            self.check(args[0], inside_aggregate=True)
            return
        if name in ("min", "max"):
            if len(args) < 2:
                self.fail(f"{name}() needs one argument (over the neighbours) or two or more (cell by cell)")
        elif name == "random":
            if args:
                self.fail("random() takes no arguments")
            self.uses_random = True
        elif name in ("abs", "floor"):
            if len(args) != 1:
                self.fail(f"{name}() takes exactly one argument")
        elif name == "round":
            if len(args) not in (1, 2):
                self.fail("round() takes one or two arguments")
        else:
            known = ", ".join(sorted(set(AGGREGATES) | set(FUNCTIONS)))
            self.fail(f"unknown function '{name}' (known: {known})")
        for arg in args:
            self.check(arg, inside_aggregate)


# ============================================================= evaluating

class EvalContext:
    """Everything an expression can read while one step is computed."""

    def __init__(self, grid: np.ndarray, views: GridViews, neighbour_offsets, dimensions: int,
                 constants: dict[str, float], state_values: dict[str, int], rng, step: int):
        self.grid = grid                          # current values (wide type, see engine.py)
        self.views = views
        self.neighbour_offsets = neighbour_offsets
        self.directions = DIRECTIONS[dimensions]
        self.constants = constants                # state names and parameters
        self.state_values = state_values          # state names only
        self.rng = rng
        self.step = step
        self._neighbours: list[np.ndarray] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.grid.shape

    def value_of(self, name: str):
        if name == "self":
            return self.grid
        if name == "step":
            return self.step
        if name in self.directions:
            return self.views.at(self.directions[name])
        return self.constants[name]

    def neighbours(self) -> list[np.ndarray]:
        """The neighbourhood, as one shifted grid per neighbour (computed once per step)."""
        if self._neighbours is None:
            self._neighbours = [self.views.at(offset) for offset in self.neighbour_offsets]
        return self._neighbours


def _evaluate(node: ast.AST, ctx: EvalContext, nb):
    """Evaluate a checked syntax tree. `nb` is the current neighbour inside an aggregate."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return nb if node.id == "nb" else ctx.value_of(node.id)
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_evaluate(node.left, ctx, nb), _evaluate(node.right, ctx, nb))
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, ctx, nb)
        if isinstance(node.op, ast.Not):
            return np.logical_not(value)
        if isinstance(node.op, ast.USub):
            return np.negative(value)
        if isinstance(node.op, ast.Invert):
            return np.invert(value)
        return value
    if isinstance(node, ast.BoolOp):
        parts = [_evaluate(part, ctx, nb) for part in node.values]   # all parts, always
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        return functools.reduce(combine, parts)
    if isinstance(node, ast.Compare):
        left, result = _evaluate(node.left, ctx, nb), None
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, ctx, nb)
            outcome = _COMPARE[type(op)](left, right)
            result = outcome if result is None else np.logical_and(result, outcome)
            left = right
        return result
    if isinstance(node, ast.IfExp):
        return np.where(_evaluate(node.test, ctx, nb), _evaluate(node.body, ctx, nb), _evaluate(node.orelse, ctx, nb))
    if isinstance(node, ast.Call):
        return _call(node, ctx, nb)
    raise TypeError(f"unsupported construct {type(node).__name__}")   # unreachable after checking


def _call(node: ast.Call, ctx: EvalContext, nb):
    name, args = node.func.id, node.args
    if name in AGGREGATES and len(args) == 1:
        return _aggregate(name, args[0], ctx)
    if name == "random":
        return ctx.rng.random(ctx.shape)
    values = [_evaluate(arg, ctx, nb) for arg in args]
    if name == "min":
        return functools.reduce(np.minimum, values)
    if name == "max":
        return functools.reduce(np.maximum, values)
    if name == "abs":
        return np.abs(values[0])
    if name == "floor":
        return np.floor(values[0])
    digits = int(values[1]) if len(values) == 2 else 0
    return np.round(values[0], digits)   # round


def _aggregate(name: str, arg: ast.expr, ctx: EvalContext):
    """count/sum/mean/min/max of `arg` evaluated once per neighbour."""
    if name == "count" and isinstance(arg, ast.Name) and arg.id in ctx.state_values:
        target = ctx.state_values[arg.id]                       # count(NAME) shorthand
        per_neighbour = [neighbour == target for neighbour in ctx.neighbours()]
    else:
        per_neighbour = [np.broadcast_to(_evaluate(arg, ctx, neighbour), ctx.shape)
                         for neighbour in ctx.neighbours()]
    stacked = np.stack(per_neighbour)          # shape: (number of neighbours, *grid shape)
    if name == "count":
        return np.count_nonzero(stacked, axis=0)
    if name == "sum":
        return stacked.sum(axis=0)
    if name == "mean":
        return stacked.mean(axis=0)
    if name == "min":
        return stacked.min(axis=0)
    return stacked.max(axis=0)
