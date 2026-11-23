"""The training code runs and learns (skipped where PyTorch is not installed).

These are small, fast checks on a toy dataset. The real training happens on a GPU;
what is checked here is that nothing in it is broken: shapes, splits, losses going
down, saving, resuming and evaluating.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cellab.dataset import build_dataset  # noqa: E402
from helpers import AUTOMATA, write_json  # noqa: E402
from ml import eca  # noqa: E402
from ml.data import load  # noqa: E402
from ml.evaluate import evaluate, evaluate_and_save  # noqa: E402
from ml.models import AnswerNet, Look, StepNet  # noqa: E402
from ml.train import train  # noqa: E402
from ml.verify import run_verification, validate_model  # noqa: E402


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ml")
    spec = {"name": "toy", "automaton": str(AUTOMATA / "elementary.json"), "size": [12],
            "settings": {"rule": {"range": [0, 255]}}, "initials": [{"type": "random", "weights": {"0": 1, "1": 1}}],
            "samples_per_setting": 10, "horizons": [1, 2, 4, 8], "max_steps": 40, "occurs_queries": 2, "seed": 1}
    return build_dataset(write_json(folder / "toy.json", spec), folder, workers=2)


def test_rule_families_and_classes():
    families = {eca.representative(rule) for rule in range(256)}
    assert len(families) == 88
    assert eca.family(30) == (30, 86, 135, 149) and eca.mirror(110) == 124 and eca.complement(110) == 137
    assert all(eca.representative(rule) == rule for rule in eca.CLASS_1 | eca.CLASS_3 | eca.CLASS_4)
    assert [eca.wolfram_class(rule) for rule in (0, 184, 30, 110, 86)] == [1, 2, 3, 4, 3]
    held = eca.held_out_families()
    assert held == eca.held_out_families() and {eca.wolfram_class(f) for f in held} == {1, 2, 3, 4}
    # mirroring a row and the rule gives the mirrored result: the families really are the same behaviour
    row = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])

    def step(rule, cells):
        return (rule >> (4 * np.roll(cells, 1) + 2 * cells + np.roll(cells, -1))) & 1
    assert (step(eca.mirror(30), row[::-1])[::-1] == step(30, row)).all()
    assert (1 - step(eca.complement(30), 1 - row) == step(30, row)).all()


def test_splits_keep_whole_families_out_of_training(dataset):
    data = load(dataset)
    held = eca.held_out_families()
    seen = set(data.rule[torch.cat([data.splits[name] for name in ("train", "val", "test")])].tolist())
    unseen = set(data.rule[data.splits["unseen"]].tolist())
    assert not seen & unseen and len(seen) + len(unseen) == 256
    assert all(eca.representative(rule) in held for rule in unseen)
    assert not any(eca.representative(rule) in held for rule in seen)
    assert sum(len(index) for index in data.splits.values()) == 2560
    assert data.cond.shape == (2560, 8) and data.values == 2 and data.wrap


def test_networks_accept_one_and_two_dimensions():
    for dims, shape in ((1, (9,)), (2, (6, 7))):
        grid = torch.randint(0, 3, (4, *shape))
        cond = torch.rand(4, 5)
        assert StepNet(dims, 3, cond=5)(grid, cond).shape == (4, 3, *shape)
        out = AnswerNet(dims, 3, cond=5, extra=2, width=16)(grid, torch.rand(4, 2, *shape), grid, cond, torch.tensor([1, 2, 4, 8]))
        assert out["state"].shape == (4, 3, *shape) and out["count"].shape == (4, 3)
        assert torch.allclose(out["count"].sum(dim=1), torch.ones(4), atol=1e-5)
        assert all(out[name].shape == (4,) for name in ("stable", "stable_at", "start", "length", "occurs"))


def test_step_network_learns_the_rules_and_simulates_unseen_ones(dataset, tmp_path):
    history = train("step", str(dataset), str(tmp_path / "step"), epochs=30, batch=128, device="cpu")
    losses = [epoch["val"]["step"] for epoch in history["epochs"]]
    assert losses[-1] < 0.02 < losses[0]
    report = evaluate("step", str(dataset), str(tmp_path / "step" / "best.pt"), device="cpu")
    for split in ("test", "unseen"):                       # 8 steps ahead by applying the learned step 8 times
        assert report["splits"][split]["all"]["by_horizon"][-1]["model"]["cells_correct"] > 0.97
    assert set(report["splits"]["unseen"]) == {"all", *eca.CLASS_NAMES.values()}


def test_answers_network_trains_resumes_and_is_evaluated(dataset, tmp_path):
    out = tmp_path / "answers"
    first = train("answers", str(dataset), str(out), epochs=2, batch=128, width=16, device="cpu")
    again = train("answers", str(dataset), str(out), epochs=3, batch=128, width=16, device="cpu", resume=True)
    assert [e["epoch"] for e in again["epochs"]] == [1, 2, 3] and again["epochs"][:2] == first["epochs"]
    assert again["epochs"][-1]["val"]["state"] < first["epochs"][0]["val"]["state"]
    assert json.loads((out / "history.json").read_text())["config"]["tasks"] == ["state", "count", "stable", "cycle", "occurs"]
    report = evaluate("answers", str(dataset), str(out / "best.pt"), device="cpu")
    scores = report["splits"]["test"]["all"]
    assert {"by_horizon", "stable", "cycle_start", "cycle_length", "occurs"} <= set(scores)
    assert 0 <= scores["occurs"]["f1"] <= 1 and scores["by_horizon"][0]["copy"]["cells_correct"] < 1

    only_count = train("answers", str(dataset), str(tmp_path / "count"), epochs=1, batch=128, width=16, device="cpu",
                       tasks=("count",), limit=256)
    assert only_count["config"]["tasks"] == ["count"]


def test_look_circular_pad_matches_index_select():
    for dims, shape in ((1, (16,)), (2, (16, 16))):
        for reach in (1, 4, 8):
            layer = Look(dims, 3, 4, wrap=True, dilation=reach)
            x = torch.randn(2, 3, *shape)
            got = layer(x)
            ref = x
            for axis in range(2, x.dim()):
                s = ref.shape[axis]
                ref = ref.index_select(axis, torch.arange(-reach, s + reach) % s)
            assert torch.allclose(got, layer.conv(ref), atol=1e-6)


def test_preloaded_data_save_every_and_verification(dataset, tmp_path):
    data = load(dataset, "cpu")
    res_dir = tmp_path / "results"
    step_dir = res_dir / "step_elementary"
    ans_dir = res_dir / "answers_elementary"
    train("step", data, str(step_dir), epochs=3, batch=128, device="cpu", save_every=2, log_every=2)
    evaluate_and_save("step", data, step_dir / "best.pt", step_dir, device="cpu")
    train("answers", data, str(ans_dir), epochs=2, batch=128, width=16, device="cpu", save_every=2, log_every=2, limit=256)
    evaluate_and_save("answers", data, ans_dir / "best.pt", ans_dir, device="cpu")

    assert (step_dir / "last.pt").exists() and (step_dir / "metrics.json").exists()
    assert (ans_dir / "last.pt").exists() and (ans_dir / "metrics.json").exists()

    v = validate_model(res_dir, "step_elementary", datasets_dir=Path("datasets"), device="cpu")
    assert v["status"] in ("PASS", "WARN", "FAIL") and v["live_sim"]

    report = run_verification(res_dir, datasets_dir=Path("datasets"), device="cpu", save=True)
    assert (res_dir / "verification.json").exists() and (res_dir / "verification_report.md").exists()
    assert report["summary"]["models_audited"] == 2
    assert report["summary"]["failed"] <= len(report["hypotheses"]) + len(report["live_simulator_checks"])
    assert len(report["speed_benchmarks"]) >= 3
    assert any(r["model"] == "step_elementary" for r in report["live_simulator_checks"])
