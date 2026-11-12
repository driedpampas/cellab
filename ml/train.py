"""Training either network.

    python -m ml.train step    data/elementary_w16.npz --out results/step
    python -m ml.train answers data/elementary_w16.npz --out results/answers
    python -m ml.train answers data/elementary_w16.npz --out results/only_count --tasks count

Method (the same for both networks): AdamW, learning rate warmed up for one epoch
and then lowered along a cosine curve, mixed precision on a GPU, fixed seeds. After
every epoch the validation loss is computed; the best model so far is kept in
best.pt and the complete training state in last.pt, so `--resume` continues an
interrupted run (Colab sessions end without warning). history.json holds the
numbers of every epoch.

The answers network learns its heads together. Each head's loss is divided by a
learned weight (Kendall et al., 2018), so no head has to be balanced by hand.
`--tasks` restricts training to some heads: that gives the "five separate
networks" the shared one is compared with.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import Data, known, load
from .models import AnswerNet, StepNet

TASKS = ("state", "count", "stable", "cycle", "occurs")


# ------------------------------------------------------------ losses

def step_loss(model: StepNet, data: Data, index: torch.Tensor) -> torch.Tensor:
    """One step: generation 0 -> generation 1 (the first horizon must be 1)."""
    return F.cross_entropy(model(data.initial[index], data.cond[index]), data.states[index, 0])


def answer_batch(data: Data, index: torch.Tensor, h: torch.Tensor, q: torch.Tensor) -> dict:
    """Inputs and targets for samples `index`, each with horizon number h and query number q."""
    return {"grid": data.initial[index], "extra": data.extra[index], "query": data.queries[index, q],
            "cond": data.cond[index], "n": data.horizons[h], "state": data.states[index, h],
            "occurs": data.occurs[index, q], "start": data.cycle_start[index],
            "length": data.cycle_length[index], "stable_from": data.stable_from[index]}


def answer_losses(model: AnswerNet, data: Data, batch: dict) -> dict[str, torch.Tensor]:
    """The loss of each head on a batch (a head with nothing to learn from in the batch gives 0)."""
    out = model(batch["grid"], batch["extra"], batch["query"], batch["cond"], batch["n"])
    target_share = F.one_hot(batch["state"], data.values).float().flatten(1, -2).mean(dim=1)

    def masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp(min=1)

    def far(prediction: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(prediction, torch.log1p(steps.clamp(min=0).float()), reduction="none")

    is_stable = known(batch["stable_from"])
    # Whether a run becomes stable is known if it did, or if its cycle (longer than 1) was found.
    stable_known = is_stable | known(batch["length"]) if data.deterministic else torch.ones_like(is_stable)
    occurs_known = known(batch["occurs"])
    return {
        "state": F.cross_entropy(out["state"], batch["state"]),
        "count": F.l1_loss(out["count"], target_share) * data.values,
        "stable": masked(F.binary_cross_entropy_with_logits(out["stable"], is_stable.float(), reduction="none"), stable_known)
                  + masked(far(out["stable_at"], batch["stable_from"]), is_stable),
        "cycle": masked(far(out["start"], batch["start"]) + far(out["length"], batch["length"]), known(batch["length"])),
        "occurs": masked(F.binary_cross_entropy_with_logits(out["occurs"], (batch["occurs"] == 1).float(), reduction="none"), occurs_known),
    }


class Weighting(torch.nn.Module):
    """Combines the heads' losses: sum of loss / weight + log(weight), with learned weights."""

    def __init__(self, tasks: tuple[str, ...]):
        super().__init__()
        self.tasks = tasks
        self.log_weight = torch.nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        chosen = torch.stack([losses[task] for task in self.tasks])
        return (chosen * torch.exp(-self.log_weight) + self.log_weight).sum()


# ------------------------------------------------------------ the loop

def build(kind: str, data: Data, width: int | None = None):
    if kind == "step":
        return StepNet(data.dims, data.values, data.cond.shape[1], width or 64, data.wrap)
    return AnswerNet(data.dims, data.values, data.cond.shape[1], data.extra.shape[1], width or 96, data.wrap)


def train(kind: str, dataset: str, out: str, epochs: int = 30, batch: int = 512, lr: float = 2e-3,
          tasks: tuple[str, ...] = TASKS, width: int | None = None, seed: int = 0, device: str | None = None,
          resume: bool = False, limit: int | None = None) -> dict:
    """Train and return the history. `limit` uses only that many training samples (for quick checks)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    data = load(dataset, device)
    if kind == "step" and (int(data.horizons[0]) != 1 or data.extra.shape[1] or not data.deterministic):
        raise SystemExit("the step network needs a deterministic automaton without layers, and horizon 1 in the dataset")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build(kind, data, width).to(device)
    train_index = data.splits["train"][:limit] if limit else data.splits["train"]
    # A head whose answer is never known in this dataset (e.g. cycles of random automata) is left out.
    has_labels = {"cycle": bool(known(data.cycle_length[train_index]).any()), "occurs": bool(known(data.occurs[train_index]).any())}
    tasks = tuple(task for task in tasks if has_labels.get(task, True))
    weighting = Weighting(tasks).to(device)
    steps_per_epoch = math.ceil(len(train_index) / batch)
    optimizer = torch.optim.AdamW([*model.parameters(), *weighting.parameters()], lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(
        (s + 1) / steps_per_epoch, 0.5 * (1 + math.cos(math.pi * min(1.0, s / (epochs * steps_per_epoch))))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    config = {"kind": kind, "dataset": Path(dataset).name, "epochs": epochs, "batch": batch, "lr": lr,
              "tasks": list(tasks), "width": width, "seed": seed, "parameters": sum(p.numel() for p in model.parameters())}
    history = {"config": config, "epochs": []}
    best = math.inf

    last = out_dir / "last.pt"
    if resume and last.exists():
        saved = torch.load(last, map_location=device)
        model.load_state_dict(saved["model"]); weighting.load_state_dict(saved["weighting"])
        optimizer.load_state_dict(saved["optimizer"]); schedule.load_state_dict(saved["schedule"])
        generator.set_state(saved["generator"])
        history, best = saved["history"], saved["best"]

    def loss_of(index: torch.Tensor, h=None, q=None) -> tuple[torch.Tensor, dict]:
        if kind == "step":
            loss = step_loss(model, data, index)
            return loss, {"step": loss}
        if h is None:       # training: a random horizon and query for every sample
            h = torch.randint(len(data.horizons), (len(index),), device=device, generator=generator)
            q = torch.randint(data.queries.shape[1], (len(index),), device=device, generator=generator)
        losses = answer_losses(model, data, answer_batch(data, index, h, q))
        return weighting(losses), losses

    @torch.no_grad()
    def validate() -> dict[str, float]:
        model.eval()
        index, totals, batches = data.splits["val"], {}, 0
        for start in range(0, len(index), 4 * batch):
            part = index[start:start + 4 * batch]
            if kind == "step":
                _, losses = loss_of(part)
            else:           # validation: every sample with a fixed horizon and query, so epochs are comparable
                h = (part % len(data.horizons)).to(device)
                _, losses = loss_of(part, h, (part % data.queries.shape[1]).to(device))
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            batches += 1
        model.train()
        return {name: value / batches for name, value in totals.items()}

    for epoch in range(len(history["epochs"]) + 1, epochs + 1):
        started, running = time.time(), 0.0
        order = train_index[torch.randperm(len(train_index), device=device, generator=generator)]
        for start in range(0, len(order), batch):
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss, _ = loss_of(order[start:start + batch])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); schedule.step()
            running += loss.item()
        val = validate()
        score = sum(val[task] for task in (tasks if kind == "answers" else ("step",)))
        history["epochs"].append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val": val,
                                  "val_score": score, "seconds": round(time.time() - started, 1)})
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "weighting": weighting.state_dict(), "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(), "generator": generator.get_state(), "history": history, "best": best}, last)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        shown = "  ".join(f"{name} {value:.4f}" for name, value in val.items())
        print(f"epoch {epoch:3d}/{epochs}  train {running / steps_per_epoch:.4f}  val: {shown}  ({time.time() - started:.0f}s)", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=("step", "answers"))
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True, help="folder for best.pt, last.pt and history.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS), help="heads to train (answers network)")
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in --out")
    parser.add_argument("--limit", type=int, help="use only this many training samples")
    args = parser.parse_args()
    train(args.kind, args.dataset, args.out, args.epochs, args.batch, args.lr, tuple(args.tasks), args.width,
          args.seed, args.device, args.resume, args.limit)


if __name__ == "__main__":
    main()
