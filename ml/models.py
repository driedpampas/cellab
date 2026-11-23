"""The two networks of the experiment. Both work in 1D and 2D and on any grid size.

StepNet      learns ONE step:  grid (and rule) -> next grid.
             A cell's next value depends only on its neighbourhood, so one 3-wide
             convolution followed by per-cell layers is exactly the right shape.
             To answer "after N steps" it is applied N times, like the simulator.

AnswerNet    answers all five questions in ONE pass, without simulating:
             grid, rule, N and a configuration C go in; five heads come out.
             Dilated convolutions let distant cells influence each other in few layers.

Comparing the two is the experiment: StepNet shows what can be learned exactly;
AnswerNet shows which answers can be had without doing the N steps.
"""

import torch
from torch import nn
from torch.nn import functional as F


def one_hot(grid: torch.Tensor, values: int) -> torch.Tensor:
    """[B, *shape] of state numbers -> [B, values, *shape] of 0/1 channels (or [B, 1, *shape] for floats)."""
    if grid.is_floating_point():
        return grid.unsqueeze(1)
    return F.one_hot(grid, values).movedim(-1, 1).float()


def spread(vector: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """[B, K] -> [B, K, *shape]: the same K numbers in every cell."""
    return vector.reshape(*vector.shape, *(1,) * (like.dim() - 2)).expand(*vector.shape, *like.shape[2:])


class Look(nn.Module):
    """A 3-wide convolution that sees beyond the grid the way the automaton does: the opposite
    side for wrap-around edges (at any distance, even further than the grid is wide), zeros otherwise."""

    def __init__(self, dims: int, inputs: int, outputs: int, wrap: bool, dilation: int = 1):
        super().__init__()
        self.conv = (nn.Conv1d if dims == 1 else nn.Conv2d)(inputs, outputs, 3, dilation=dilation)
        self.wrap, self.reach = wrap, dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.wrap or all(size >= self.reach for size in x.shape[2:]):
            pad = (self.reach, self.reach) * (x.dim() - 2)
            return self.conv(F.pad(x, pad, mode="circular" if self.wrap else "constant"))
        for axis in range(2, x.dim()):
            size = x.shape[axis]
            x = x.index_select(axis, torch.arange(-self.reach, size + self.reach, device=x.device) % size)
        return self.conv(x)


class StepNet(nn.Module):
    def __init__(self, dims: int, values: int, cond: int = 0, width: int = 64, wrap: bool = True):
        super().__init__()
        conv = nn.Conv1d if dims == 1 else nn.Conv2d
        self.values = values
        self.look = Look(dims, values + cond, width, wrap)
        self.think = nn.Sequential(nn.ReLU(), conv(width, width, 1), nn.ReLU(), conv(width, width, 1), nn.ReLU(),
                                   conv(width, values, 1))

    def forward(self, grid: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Scores for every cell's next value: [B, values, *shape]."""
        x = one_hot(grid, self.values)
        return self.think(self.look(torch.cat([x, spread(cond, x)], dim=1)))

    @torch.no_grad()
    def rollout(self, grid: torch.Tensor, cond: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply the learned step `steps` times, always continuing from the most likely grid."""
        for _ in range(steps):
            out = self(grid, cond)
            grid = out.squeeze(1) if grid.is_floating_point() else out.argmax(dim=1)
        return grid


class _Block(nn.Module):
    """One residual block: look around (dilated), mix per cell, adjusted by what the model was told."""

    def __init__(self, dims: int, width: int, dilation: int, wrap: bool, told: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.adjust = nn.Linear(told, 2 * width)
        self.look = Look(dims, width, width, wrap, dilation)
        self.mix = (nn.Conv1d if dims == 1 else nn.Conv2d)(width, width, 1)

    def forward(self, h: torch.Tensor, told: torch.Tensor) -> torch.Tensor:
        scale, shift = spread(self.adjust(told), h).chunk(2, dim=1)
        y = self.norm(h) * (1 + scale) + shift
        return h + self.mix(F.relu(self.look(F.relu(y))))


class AnswerNet(nn.Module):
    """Shared encoder, five heads (the "single pass" design of Deliverable 1).

    forward(...) returns a dict:
        state      [B, values, *shape]  scores for each cell's value after N steps
        count      [B, values]          share of cells holding each value after N steps
        stable     [B]                  score for "it reaches a stable state"
        stable_at  [B]                  log(1 + step from which it is stable)
        start      [B]                  log(1 + step at which the cycle starts)
        length     [B]                  log(1 + cycle length)
        occurs     [B]                  score for "configuration C occurs"
    """

    DILATIONS = (1, 2, 4, 8, 1, 2, 4, 8)

    def __init__(self, dims: int, values: int, cond: int = 0, extra: int = 0, width: int = 96, wrap: bool = True,
                 dropout: float = 0.15):
        super().__init__()
        conv = nn.Conv1d if dims == 1 else nn.Conv2d
        self.values = values
        self.told = nn.Sequential(nn.Linear(cond + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.enter = Look(dims, 2 * values + extra, width, wrap)
        self.blocks = nn.ModuleList(_Block(dims, width, d, wrap, 64) for d in self.DILATIONS)
        self.state = conv(width, values, 1)
        self.summary = nn.Sequential(nn.Linear(2 * width + 64, 128), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.count = nn.Linear(128, values)
        self.numbers = nn.Linear(128, 5)           # stable, stable_at, start, length, occurs

    def forward(self, grid, extra, query, cond, n) -> dict[str, torch.Tensor]:
        told = self.told(torch.cat([cond, torch.log1p(n.float()).unsqueeze(1) / 4], dim=1))
        h = self.enter(torch.cat([one_hot(grid, self.values), one_hot(query, self.values), extra], dim=1))
        for block in self.blocks:
            h = block(h, told)
        flat = h.flatten(2)
        summary = self.drop(self.summary(torch.cat([flat.mean(dim=2), flat.amax(dim=2), told], dim=1)))
        stable, stable_at, start, length, occurs = self.numbers(summary).unbind(dim=1)
        count = self.count(summary) if grid.is_floating_point() else F.softmax(self.count(summary), dim=1)
        return {"state": self.state(h), "count": count, "stable": stable,
                "stable_at": stable_at, "start": start, "length": length, "occurs": occurs}
