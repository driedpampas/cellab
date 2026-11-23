"""Loading a dataset for training: tensors on the device, and the train / validation / test splits.

Splits
    Every dataset: of each setting's samples, 80% train, 10% validation, 10% test (by sample number).
    Elementary rules only: 15% of the rule families of every Wolfram class are held out
    completely. Their samples form the split "unseen", which answers "does the model work
    for rules it never saw?".

Encoding (done inside the models): each state value becomes its own 0/1 channel; layers
(energy, heading, ...) become extra channels scaled to 0..1; an elementary rule is given
as its 8 table bits; the number of steps N as log(1 + N).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cellab.dataset import UNKNOWN, load_dataset

from . import eca


@dataclass
class Data:
    name: str
    dims: int                    # 1 or 2
    values: int                  # number of state values = input channels of the grid
    wrap: bool                   # wrap-around edges (circular padding) or fixed (zero padding)
    deterministic: bool
    horizons: torch.Tensor       # [H] the N of "after N steps"
    initial: torch.Tensor        # [S, *shape] generation 0 (cell states)
    extra: torch.Tensor          # [S, L, *shape] the layers of generation 0, scaled to 0..1 (L may be 0)
    states: torch.Tensor         # [S, H, *shape]
    queries: torch.Tensor        # [S, Q, *shape]
    occurs: torch.Tensor         # [S, Q] 1 / 0 / -1 unknown
    cycle_start: torch.Tensor    # [S] or -1
    cycle_length: torch.Tensor   # [S] or -1
    stable_from: torch.Tensor    # [S] or -1
    cond: torch.Tensor           # [S, K] what else the model is told (the rule's 8 bits; K may be 0)
    group: torch.Tensor          # [S] a number to report results by (the Wolfram class, or 0)
    rule: torch.Tensor           # [S] the rule number (or 0)
    splits: dict[str, torch.Tensor]   # split name -> sample numbers
    continuous: bool = False     # True for decimal-valued automata (e.g. diffusion)
    scale: float = 1.0           # divisor used to normalize continuous values into 0..1

    @property
    def cells(self) -> int:
        return int(np.prod(self.initial.shape[1:]))


def load(path: str | Path, device: str = "cpu") -> Data:
    arrays, meta = load_dataset(path)
    info, spec = meta["automaton"], meta["spec"]
    continuous = arrays["states"].dtype.kind == "f"
    layers = len(info["layers"])
    initial = arrays["initial"][:, 0] if layers else arrays["initial"]
    extra = arrays["initial"][:, 1:].astype(np.float32) if layers else np.zeros((len(initial), 0, *initial.shape[1:]), np.float32)
    if layers:
        extra /= np.maximum(extra.max(axis=tuple(range(2, extra.ndim)), keepdims=True).max(axis=0, keepdims=True), 1)
    if continuous:
        lowest, values = 0, 1
        scale = float(max(initial.max(), arrays["states"].max(), arrays["queries"].max(), 1.0))
    else:
        # Automata whose values cannot be listed (whole numbers without a maximum) use the range seen in the data.
        seen = [initial, arrays["states"], arrays["queries"]]
        lowest = info["values"][0] if info["values"] else int(min(a.min() for a in seen))
        values = len(info["values"]) if info["values"] else int(max(a.max() for a in seen)) - lowest + 1
        scale = 1.0

    samples = len(initial)
    setting = arrays["setting"]
    number_in_setting = np.arange(samples) % spec["samples_per_setting"]
    part = np.where(number_in_setting % 10 == 8, 1, np.where(number_in_setting % 10 == 9, 2, 0))   # 0 train, 1 val, 2 test
    rules = np.array([s.get("rule", 0) for s in meta["settings"]])[setting]
    if "rule" in spec["settings"]:
        cond = np.stack([eca.rule_bits(rule) for rule in rules]).astype(np.float32)
        group = np.array([eca.wolfram_class(rule) for rule in rules])
        held = eca.held_out_families()
        unseen = np.array([eca.representative(rule) in held for rule in rules])
    else:
        cond, group, unseen = np.zeros((samples, 0), np.float32), np.zeros(samples, int), np.zeros(samples, bool)
    splits = {name: np.flatnonzero((part == k) & ~unseen) for k, name in enumerate(("train", "val", "test"))}
    if unseen.any():
        splits["unseen"] = np.flatnonzero(unseen)

    def tensor(array, dtype=torch.long):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

    if continuous:
        init_t = tensor(initial.astype(np.float32) / scale, torch.float32)
        states_t = tensor(arrays["states"].astype(np.float32) / scale, torch.float32)
        queries_t = tensor(arrays["queries"].astype(np.float32) / scale, torch.float32)
    else:
        init_t = tensor(initial.astype(np.int64) - lowest)
        states_t = tensor(arrays["states"].astype(np.int64) - lowest)
        queries_t = tensor(arrays["queries"].astype(np.int64) - lowest)

    return Data(
        name=meta["name"], dims=info["dimensions"], values=values, wrap=info["edges"] == "wrap",
        deterministic=info["deterministic"], horizons=tensor(spec["horizons"]),
        initial=init_t, extra=tensor(extra, torch.float32),
        states=states_t, queries=queries_t,
        occurs=tensor(arrays["occurs"]), cycle_start=tensor(arrays["cycle_start"]),
        cycle_length=tensor(arrays["cycle_length"]), stable_from=tensor(arrays["stable_from"]),
        cond=tensor(cond, torch.float32), group=tensor(group), rule=tensor(rules),
        splits={name: tensor(index) for name, index in splits.items()},
        continuous=continuous, scale=scale,
    )


def known(values: torch.Tensor) -> torch.Tensor:
    return values != UNKNOWN
