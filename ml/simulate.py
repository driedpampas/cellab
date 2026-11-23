"""All five answers by simulating with the learned step (StepNet), instead of skipping the simulation.

    python -m ml.simulate data/elementary_w16.npz results/step_elementary/best.pt --out results/simulated_elementary

AnswerNet answers in one pass and cannot know which cycle a chaotic row falls into: for
Rule 30 on 16 cells there are only 5 cycle lengths (1, 8, 40, 4144, 6016), but which one a
row ends in is only found by running it. StepNet learned the step exactly, so running it
until a row repeats gives the cycle exactly, like the simulator does.

How (for each sample, all on the GPU at once):
    cycle      Brent's algorithm on the rollout: the length L, then the first step S of the cycle
               (the dataset's cycle_start and cycle_length).
    stable     L == 1; it is then stable from step S (the dataset's stable_from).
    occurs     a second rollout of S + L steps, which visits every row the run will ever hold,
               compared with each query.
    state      the rows after each N, recorded during that second rollout.
The answers are written as metrics.json in the same format as an AnswerNet's (same splits,
Wolfram classes and baselines), so the two can be compared head to head.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from . import eca
from .data import Data, load
from .evaluate import score, shares
from .train import build


class Stepper:
    """The learned step as a function grid -> next grid, in full fp32 precision."""

    def __init__(self, model, cond: torch.Tensor):
        self.model, self.cond = model, cond

    def __call__(self, grid: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return self.model(grid, self.cond[rows]).argmax(dim=1)


def _same(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a == b).flatten(1).all(dim=1)


@torch.no_grad()
def find_cycles(step: Stepper, x0: torch.Tensor, max_steps: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Brent's algorithm for every row of x0 at once. Returns (start, length, forward passes);
    -1 where no cycle was found within about max_steps steps. Rows that are done drop out."""
    count, device = len(x0), x0.device
    length = torch.full((count,), -1, dtype=torch.long, device=device)
    start = torch.full((count,), -1, dtype=torch.long, device=device)
    passes = 0

    # Phase 1: the cycle length. The tortoise waits at powers of two while the hare runs ahead.
    rows = torch.arange(count, device=device)
    tortoise, hare = x0.clone(), step(x0, rows)
    power = torch.ones(count, dtype=torch.long, device=device)
    lam = torch.ones(count, dtype=torch.long, device=device)
    passes += 1
    for _ in range(2 * max_steps):
        hit = _same(tortoise, hare)
        if hit.any():
            length[rows[hit]] = lam[hit]
            keep = ~hit
            rows, tortoise, hare, power, lam = rows[keep], tortoise[keep], hare[keep], power[keep], lam[keep]
        if not len(rows):
            break
        reset = power == lam
        tortoise = torch.where(reset.view(-1, *(1,) * (x0.dim() - 1)), hare, tortoise)
        power = torch.where(reset, power * 2, power)
        lam = torch.where(reset, torch.zeros_like(lam), lam)
        hare = step(hare, rows)
        lam += 1
        passes += 1

    # Phase 2: the start. A hare L steps ahead and a tortoise from the start meet where the cycle begins.
    rows = torch.nonzero(length >= 0).flatten()
    if len(rows):
        ahead, hare = length[rows], x0[rows].clone()
        for k in range(int(ahead.max())):
            moving = torch.nonzero(ahead > k).flatten()
            hare[moving] = step(hare[moving], rows[moving])
            passes += 1
        tortoise, mu = x0[rows].clone(), torch.zeros(len(rows), dtype=torch.long, device=device)
        active = torch.arange(len(rows), device=device)
        while len(active):
            hit = _same(tortoise[active], hare[active])
            active = active[~hit]
            if not len(active):
                break
            tortoise[active] = step(tortoise[active], rows[active])
            hare[active] = step(hare[active], rows[active])
            mu[active] += 1
            passes += 2
        start[rows] = mu
    return start, length, passes


@torch.no_grad()
def visit(step: Stepper, x0: torch.Tensor, queries: torch.Tensor, horizons: list[int], until: torch.Tensor):
    """Roll out every row for max(until, largest N) steps: the rows after each N, and whether each
    query appears among the first `until` rows (steps 0 .. until-1)."""
    count = len(x0)
    states = torch.empty((count, len(horizons), *x0.shape[1:]), dtype=x0.dtype, device=x0.device)
    seen = torch.zeros(queries.shape[:2], dtype=torch.bool, device=x0.device)
    grid, rows = x0.clone(), torch.arange(count, device=x0.device)
    last = max(int(until.max()) if count else 0, horizons[-1])
    where = {n: h for h, n in enumerate(horizons)}
    for k in range(last + 1):
        looking = until > k
        if looking.any():
            for q in range(queries.shape[1]):
                seen[:, q] |= looking & _same(grid, queries[:, q])
        if k in where:
            states[:, where[k]] = grid
        if k < last:
            grid = step(grid, rows)
    return states, seen


def simulate(data: Data, checkpoint: str | Path, max_steps: int | None = None, batch: int = 8192,
             device: str | None = None) -> dict:
    """The report (metrics.json contents) of StepNet + simulation on the test and unseen splits."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device)
    if saved["config"]["kind"] != "step":
        raise SystemExit("ml.simulate needs a StepNet checkpoint (kind 'step')")
    if data.continuous or not data.deterministic or data.extra.shape[1]:
        raise SystemExit("ml.simulate needs a deterministic automaton with whole-number states and no layers")
    model = build("step", data, saved["config"]["width"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    if max_steps is None:
        # As far as the dataset looked; for small grids at least the number of possible grids, so every cycle is found.
        max_steps = max(int(data.cycle_start.max() + data.cycle_length.max()) + 1, 64)
        if data.values ** data.cells <= 1 << 20:
            max_steps = max(max_steps, data.values ** data.cells)
    horizons = data.horizons.tolist()

    # Exact fp32: TF32 could flip a borderline cell, and one wrong cell changes a whole cycle.
    tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    report = {"kind": "simulated", "dataset": data.name, "trained_on": saved["config"]["dataset"], "epoch": saved["epoch"],
              "parameters": saved["config"]["parameters"], "method": "StepNet rollout + Brent cycle detection",
              "max_steps": max_steps, "splits": {}}
    try:
        started, total_passes = time.time(), 0
        for split in ("test", "unseen"):
            if split not in data.splits:
                continue
            index = data.splits[split]
            parts: dict[str, list] = {}
            for begin in range(0, len(index), batch):
                part = index[begin:begin + batch]
                step = Stepper(model, data.cond[part])
                x0 = data.initial[part]
                start, length, passes = find_cycles(step, x0, max_steps)
                found = length >= 0
                until = torch.where(found, start + length, torch.full_like(start, max_steps))
                states, seen = visit(step, x0, data.queries[part], horizons, until)
                total_passes += passes + int(until.max()) + 1
                stable = found & (length == 1)
                big = float(max_steps)
                parts.setdefault("state", []).append(states)
                parts.setdefault("count", []).append(torch.stack([shares(states[:, h], data.values) for h in range(len(horizons))], dim=1))
                parts.setdefault("stable", []).append(torch.where(stable, 1.0, -1.0))
                parts.setdefault("stable_at", []).append(torch.log1p(torch.where(stable, start, 0).double()).float())
                parts.setdefault("start", []).append(torch.log1p(torch.where(found, start.double(), big)).float())
                parts.setdefault("length", []).append(torch.log1p(torch.where(found, length.double(), big)).float())
                # Not seen in a finished run: certainly never (score >0 means yes).
                parts.setdefault("occurs", []).append(torch.where(seen, 1.0, -1.0))
            said = {name: torch.cat(values) for name, values in parts.items()}
            groups = {"all": score("answers", data, index, said, data.splits["train"])}
            for group in sorted(set(data.group[index].tolist()) - {0}):
                mask = data.group[index] == group
                groups[eca.CLASS_NAMES[group]] = score("answers", data, index[mask], {k: v[mask] for k, v in said.items()},
                                                       data.splits["train"])
            report["splits"][split] = groups
        report["seconds"] = round(time.time() - started, 2)
        report["forward_passes"] = total_passes
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    return report


def simulate_and_save(data: Data | str | Path, checkpoint: str | Path, out: str | Path, max_steps: int | None = None,
                      device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = data if isinstance(data, Data) else load(data, device)
    report = simulate(data, checkpoint, max_steps, device=device)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, groups in report["splits"].items():
        everything = groups["all"]
        line = [f"{split}:"]
        if "cycle_length" in everything:
            line.append(f"cycle length exact {everything['cycle_length']['exactly_right']:.4f}")
            line.append(f"start exact {everything['cycle_start']['exactly_right']:.4f}")
        for key in ("stable", "occurs"):
            if key in everything:
                line.append(f"{key} F1 {everything[key]['f1']:.4f}")
        line.append(f"cells N={everything['by_horizon'][-1]['n']} {everything['by_horizon'][-1]['model']['cells_correct']:.4f}")
        print("  ".join(line), flush=True)
    print(f"({report['forward_passes']:,} StepNet passes in {report['seconds']:.1f}s, max_steps {report['max_steps']:,})", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("checkpoint", help="a StepNet best.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, help="give up on a cycle after about this many steps (default: enough for the dataset)")
    parser.add_argument("--limit", type=int, help="only the first this many samples of each split (quick checks)")
    parser.add_argument("--device")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load(args.dataset, device)
    if args.limit:
        data.splits = {name: index if name == "train" else index[:args.limit] for name, index in data.splits.items()}
    simulate_and_save(data, args.checkpoint, args.out, args.max_steps, device)


if __name__ == "__main__":
    main()
