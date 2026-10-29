"""Seeded random numbers: the same seed always gives the same run.

A run owns exactly one generator, created from its seed. Nothing in CelLab uses
Python's `random` module or NumPy's global random functions (a test checks this),
so the only randomness in a run comes from this generator, drawn in a fixed order:

    1. the random initial world (if the experiment asks for one)
    2. then, every step, one full-grid array per random() in the rules, in file order

The generator's internal state is a small dictionary. It is saved in every
checkpoint, so a resumed run continues the exact same random sequence.
"""

import json
import secrets

import numpy as np


def new_seed() -> int:
    """A fresh seed for runs whose experiment file does not give one (it is recorded)."""
    return secrets.randbelow(2**32)


def make_generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def generator_state(rng: np.random.Generator) -> str:
    """The generator's full internal state, as JSON text."""
    return json.dumps(rng.bit_generator.state)


def restore_generator(state: str) -> np.random.Generator:
    """A generator that continues exactly where the saved one stopped."""
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = json.loads(state)
    return rng
