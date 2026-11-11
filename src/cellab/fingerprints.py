"""Fingerprints (hashes) of grids, and the history of a run built from them.

A fingerprint is a 16-byte BLAKE2b hash of a grid's contents, shape and number
type. Two different grids get the same fingerprint with a probability of about
1 in 2^128, so in practice equal fingerprints mean equal grids, and comparing 16
bytes is much cheaper than comparing whole grids.

The History remembers the fingerprint of every step and the first step at which
each fingerprint appeared. That is enough to answer three of the five questions:

    the first repeat      fingerprint(t) was already seen at step s  ->  cycle from s, length t - s
    the first standstill  fingerprint(t) == fingerprint(t - 1)       ->  stable from t - 1
    "does C occur?"       look C's fingerprint up in `grid_seen`

For automata with layers a generation is more than its grid (an agent's energy or
heading also counts for "is this the same state?"). Each step then gets two
fingerprints: one of the whole state, used for repeats and standstills, and one of
the grid alone, used for "does this configuration occur?". Without layers the two
are the same.
"""

import hashlib

import numpy as np

FINGERPRINT_SIZE = 16   # bytes


def fingerprint(grid: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=FINGERPRINT_SIZE)
    h.update(f"{grid.shape}|{grid.dtype.str}|".encode())
    h.update(np.ascontiguousarray(grid).tobytes())
    return h.digest()


class History:
    """Fingerprints of every step of a run, step 0 included."""

    def __init__(self):
        self.fingerprints: list[bytes] = []          # index = step number
        self.first_seen: dict[bytes, int] = {}       # fingerprint -> first step with it
        self.grid_seen: dict[bytes, int] = {}        # fingerprint of the grid alone -> first step with it
        self.repeat: tuple[int, int] | None = None   # (earlier step, later step) of the first repeat
        self.standstill: int | None = None           # first step t with grid(t) == grid(t - 1)

    @property
    def last_step(self) -> int:
        return len(self.fingerprints) - 1

    @property
    def cycle(self) -> tuple[int, int] | None:
        """(first step of the cycle, cycle length), once a grid has repeated."""
        if self.repeat is None:
            return None
        earlier, later = self.repeat
        return earlier, later - earlier

    def record(self, fp: bytes, grid_fp: bytes | None = None) -> None:
        """Add the fingerprint of the next step (and of its grid alone, if that differs)."""
        step = len(self.fingerprints)
        self.grid_seen.setdefault(fp if grid_fp is None else grid_fp, step)
        if fp in self.first_seen:
            if self.repeat is None:
                self.repeat = (self.first_seen[fp], step)
        else:
            self.first_seen[fp] = step
        if self.standstill is None and self.fingerprints and fp == self.fingerprints[-1]:
            self.standstill = step
        self.fingerprints.append(fp)

    @classmethod
    def rebuild(cls, records: list[bytes]) -> "History":
        """Recreate a History from saved records (used when resuming). A record is one
        fingerprint, or two in a row when the automaton has layers."""
        history = cls()
        for record in records:
            history.record(record[:FINGERPRINT_SIZE], record[FINGERPRINT_SIZE:] or None)
        return history
