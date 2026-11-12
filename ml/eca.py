"""Facts about the 256 elementary rules that the experiments group their results by.

Rule number -> table: bit k of the number is the new value for the pattern whose three
cells (left, self, right) read as the binary number k. Rule 30 = 00011110: pattern 100 -> 1 ...

Two rules are *equivalent* if one becomes the other by swapping left and right
(mirror), by swapping 0 and 1 (complement), or both: they behave identically up to
that relabelling. The 256 rules fall into 88 such families. A fair "rules never
seen in training" test must hold out whole families, otherwise the model has seen
the mirror image of every test rule.

Wolfram's four classes (the usual table for the 88 families, by smallest member):
1 dies to a uniform state, 2 settles into stripes or short cycles, 3 chaotic,
4 complex localised structures.
"""

import numpy as np

CLASS_1 = {0, 8, 32, 40, 128, 136, 160, 168}
CLASS_3 = {18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150}
CLASS_4 = {41, 54, 106, 110}
CLASS_NAMES = {1: "1 uniform", 2: "2 periodic", 3: "3 chaotic", 4: "4 complex"}


def rule_bits(rule: int) -> np.ndarray:
    """The rule's table: bits[k] is the new value for the pattern with number k = 4*left + 2*self + right."""
    return np.array([(rule >> k) & 1 for k in range(8)], dtype=np.uint8)


def mirror(rule: int) -> int:
    """The rule that does the same with left and right swapped."""
    return sum(((rule >> (4 * r + 2 * c + l)) & 1) << (4 * l + 2 * c + r) for l in (0, 1) for c in (0, 1) for r in (0, 1))


def complement(rule: int) -> int:
    """The rule that does the same with 0 and 1 swapped."""
    return sum((1 - ((rule >> (7 - k)) & 1)) << k for k in range(8))


def family(rule: int) -> tuple[int, ...]:
    return tuple(sorted({rule, mirror(rule), complement(rule), mirror(complement(rule))}))


def representative(rule: int) -> int:
    """The smallest rule of the family: the name the family goes by."""
    return family(rule)[0]


def wolfram_class(rule: int) -> int:
    smallest = representative(rule)
    return 1 if smallest in CLASS_1 else 3 if smallest in CLASS_3 else 4 if smallest in CLASS_4 else 2


def held_out_families(share: float = 0.15, seed: int = 0) -> set[int]:
    """Families (by representative) kept out of training, the same share of every class (at least one)."""
    rng = np.random.default_rng(seed)
    families = sorted({representative(rule) for rule in range(256)})
    held = set()
    for cls in (1, 2, 3, 4):
        members = [f for f in families if wolfram_class(f) == cls]
        held.update(rng.choice(members, size=max(1, round(share * len(members))), replace=False).tolist())
    return held
