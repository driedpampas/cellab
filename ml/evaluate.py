"""Measuring a trained network, next to simple baselines.

    python -m ml.evaluate step    data/elementary_w16.npz results/step/best.pt    --out results/step
    python -m ml.evaluate answers data/elementary_w16.npz results/answers/best.pt --out results/answers
    python -m ml.evaluate step    data/elementary_w32_test.npz results/step/best.pt --out results/step_w32

Writes metrics.json (and, if matplotlib is installed, accuracy_by_horizon.png). Everything
is reported for the test split (rules seen in training, runs not seen) and, for the
elementary rules, for the "unseen" split (rule families never seen), each also broken
down by Wolfram class and by the number of steps N.

Metrics, as announced in Deliverable 1
    state after N    share of cells correct; share of grids completely correct
    count after N    average absolute error, in cells
    stable?          accuracy and F1; average error of the step (where it is stable)
    cycle            average error of the start and of the length; share of lengths exactly right
    does C occur?    accuracy and F1

Baselines
    copy    answer with the starting grid (state and count)
    common  answer with the most common answer of the training split
            (state: each cell's most common value; numbers: the median)
"""

import argparse
import json
from pathlib import Path

import torch

from . import eca
from .data import Data, known, load
from .models import one_hot
from .train import answer_batch, build


def grid_scores(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    right = (prediction == target).flatten(1).float().mean(dim=1)
    return {"cells_correct": right.mean().item(), "grids_correct": (right == 1).float().mean().item()}


def yes_no_scores(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    hits = (prediction & target).sum().item()
    precision, recall = hits / max(prediction.sum().item(), 1), hits / max(target.sum().item(), 1)
    return {"accuracy": (prediction == target).float().mean().item(), "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "share_yes": target.float().mean().item()}


def shares(grid: torch.Tensor, values: int) -> torch.Tensor:
    return one_hot(grid, values).flatten(2).mean(dim=2)


@torch.no_grad()
def predictions(kind: str, model, data: Data, index: torch.Tensor, batch: int = 2048) -> dict[str, torch.Tensor]:
    """Everything the model says about samples `index`: per horizon [n, H, ...] and per query [n, Q]."""
    horizons, queries = len(data.horizons), data.queries.shape[1]
    parts: dict[str, list] = {}

    def keep(name, value):
        parts.setdefault(name, []).append(value)

    for start in range(0, len(index), batch):
        part = index[start:start + batch]
        if kind == "step":
            grid, states, done = data.initial[part], [], 0
            for n in data.horizons.tolist():
                grid = model.rollout(grid, data.cond[part], n - done)
                states.append(grid)
                done = n
            keep("state", torch.stack(states, dim=1))
            continue
        per_h = [model(**{k: v for k, v in answer_batch(data, part, torch.full_like(part, h), torch.zeros_like(part)).items()
                          if k in ("grid", "extra", "query", "cond", "n")}) for h in range(horizons)]
        keep("state", torch.stack([out["state"].argmax(dim=1) for out in per_h], dim=1))
        keep("count", torch.stack([out["count"] for out in per_h], dim=1))
        middle = per_h[horizons // 2]           # the whole-run answers do not depend on N: read them at one N
        for name in ("stable", "stable_at", "start", "length"):
            keep(name, middle[name])
        last_n = torch.full_like(part, horizons // 2)
        keep("occurs", torch.stack([model(**{k: v for k, v in answer_batch(data, part, last_n, torch.full_like(part, q)).items()
                                             if k in ("grid", "extra", "query", "cond", "n")})["occurs"] for q in range(queries)], dim=1))
    result = {name: torch.cat(values) for name, values in parts.items()}
    if kind == "step":
        result["count"] = torch.stack([shares(result["state"][:, h], data.values) for h in range(horizons)], dim=1)
    return result


def score(kind: str, data: Data, index: torch.Tensor, said: dict, train: torch.Tensor) -> dict:
    """All metrics for samples `index` (said = predictions for exactly those samples)."""
    cells, result = data.cells, {"samples": len(index)}
    if not len(index):
        return result
    target_states = data.states[index]
    target_share = torch.stack([shares(target_states[:, h], data.values) for h in range(len(data.horizons))], dim=1)
    common_grid = data.states[train].flatten(0, 1).mode(dim=0).values
    by_horizon = []
    for h, n in enumerate(data.horizons.tolist()):
        truth = target_states[:, h]
        row = {"n": n, "model": grid_scores(said["state"][:, h], truth), "copy": grid_scores(data.initial[index], truth),
               "common": grid_scores(common_grid.expand_as(truth), truth)}
        row["model"]["count_error"] = (said["count"][:, h] - target_share[:, h]).abs().sum(dim=1).mean().item() * cells / 2
        row["copy"]["count_error"] = (shares(data.initial[index], data.values) - target_share[:, h]).abs().sum(dim=1).mean().item() * cells / 2
        by_horizon.append(row)
    result["by_horizon"] = by_horizon
    if kind == "step":
        return result

    def steps(prediction: torch.Tensor) -> torch.Tensor:
        return torch.expm1(prediction.float()).round().clamp(min=0)

    def number_scores(prediction, truth, baseline) -> dict:
        return {"error": (steps(prediction) - truth).abs().mean().item(), "exactly_right": (steps(prediction) == truth).float().mean().item(),
                "common_error": (baseline - truth).abs().mean().item(), "samples": len(truth)}

    is_stable = known(data.stable_from[index])
    stable_known = is_stable | known(data.cycle_length[index]) if data.deterministic else torch.ones_like(is_stable)
    if stable_known.any():
        result["stable"] = yes_no_scores(said["stable"][stable_known] > 0, is_stable[stable_known])
        if is_stable.any():
            result["stable"]["step"] = number_scores(said["stable_at"][is_stable], data.stable_from[index][is_stable].float(),
                                                     data.stable_from[train][known(data.stable_from[train])].float().median())
    cycle_known = known(data.cycle_length[index])
    if cycle_known.any():
        seen = known(data.cycle_length[train])
        for name, values in (("start", data.cycle_start), ("length", data.cycle_length)):
            result[f"cycle_{name}"] = number_scores(said[name][cycle_known], values[index][cycle_known].float(),
                                                    values[train][seen].float().median())
    occurs_known = known(data.occurs[index])
    if occurs_known.any():
        result["occurs"] = yes_no_scores(said["occurs"][occurs_known] > 0, data.occurs[index][occurs_known] == 1)
    return result


def evaluate(kind: str, dataset: str, checkpoint: str, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load(dataset, device)
    saved = torch.load(checkpoint, map_location=device)
    model = build(kind, data, saved["config"]["width"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    report = {"kind": kind, "dataset": data.name, "trained_on": saved["config"]["dataset"], "epoch": saved["epoch"],
              "parameters": saved["config"]["parameters"], "splits": {}}
    for split in ("test", "unseen"):
        if split not in data.splits:
            continue
        index = data.splits[split]
        said = predictions(kind, model, data, index)
        part = {"all": score(kind, data, index, said, data.splits["train"])}
        for group in sorted(set(data.group[index].tolist()) - {0}):
            mask = data.group[index] == group
            part[eca.CLASS_NAMES[group]] = score(kind, data, index[mask], {k: v[mask] for k, v in said.items()}, data.splits["train"])
        report["splits"][split] = part
    return report


def plot(report: dict, path: Path) -> None:
    """Cells correct against the number of steps, one line per Wolfram class, for each split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = report["splits"]
    figure, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for axis, (split, groups) in zip(axes[0], splits.items()):
        for name, scores in groups.items():
            if "by_horizon" in scores:
                axis.plot([row["n"] for row in scores["by_horizon"]], [row["model"]["cells_correct"] for row in scores["by_horizon"]],
                          marker="o", label=name, linewidth=3 if name == "all" else 1.5)
        copy = groups["all"]["by_horizon"]
        axis.plot([row["n"] for row in copy], [row["copy"]["cells_correct"] for row in copy], "k--", label="copy the start")
        axis.set_xscale("log", base=2); axis.set_ylim(0.4, 1.02)
        axis.set_xlabel("steps ahead (N)"); axis.set_ylabel("cells correct")
        axis.set_title(f"{report['kind']} network on {report['dataset']}: {split} split"); axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=("step", "answers"))
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = evaluate(args.kind, args.dataset, args.checkpoint, args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, groups in report["splits"].items():
        rows = groups["all"].get("by_horizon", [])
        print(f"{split}: " + "  ".join(f"N={row['n']}: {row['model']['cells_correct']:.3f}" for row in rows))
    try:
        plot(report, out / "accuracy_by_horizon.png")
    except ImportError:
        print("(matplotlib is not installed: no picture)")


if __name__ == "__main__":
    main()
