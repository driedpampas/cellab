"""Post-training verification and inference benchmarks.

    python -m ml.verify results/ --data data --datasets datasets

Checks every trained model in `results/` for completeness, convergence and finite
metrics; verifies the core Deliverable 3 hypotheses (StepNet exactness & size
generalization, AnswerNet Wolfram-class ordering, multi-task vs single-task);
cross-checks trained StepNet checkpoints live against `cellab.engine.step` on
fresh grids; and benchmarks inference speed (simulator vs StepNet vs AnswerNet).
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cellab.automaton import load_automaton
from cellab.dataset import load_spec
from cellab.engine import step as engine_step
from cellab.randomness import make_generator

from . import eca
from .models import AnswerNet, StepNet


def _is_finite_tree(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_is_finite_tree(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_is_finite_tree(v) for v in obj)
    return True


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def audit_run(folder: Path) -> dict:
    """Audit one model directory containing metrics.json for integrity and baseline deltas."""
    metrics_path = folder / "metrics.json"
    name = folder.name
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    finite = _is_finite_tree(report)
    history_path = folder / "history.json"
    epochs_done, best_epoch, total_s, converged = None, report.get("epoch"), None, None
    if history_path.exists():
        hist = json.loads(history_path.read_text(encoding="utf-8"))
        finite = finite and _is_finite_tree(hist)
        ep = hist.get("epochs", [])
        epochs_done = len(ep)
        if ep:
            total_s = round(sum(e.get("seconds", 0.0) for e in ep), 1)
            best_val = min(e["val_score"] for e in ep)
            converged = best_val <= ep[0]["val_score"] + 1e-9

    test_all = report.get("splits", {}).get("test", {}).get("all", {})
    horizons = test_all.get("by_horizon", [])
    n1 = horizons[0] if horizons else None
    nlast = horizons[-1] if horizons else None
    cells_n1 = n1["model"]["cells_correct"] if n1 else None
    cells_last = nlast["model"]["cells_correct"] if nlast else None
    grids_last = nlast["model"]["grids_correct"] if nlast else None
    copy_last = nlast["copy"]["cells_correct"] if nlast else None
    common_last = nlast["common"]["cells_correct"] if nlast else None
    best_base = max(copy_last, common_last) if (copy_last is not None and common_last is not None) else None

    status = "PASS"
    notes = []
    if not finite:
        status = "FAIL"; notes.append("non-finite values in metrics/history")
    if converged is False:
        status = "WARN"; notes.append("val_score never improved over epoch 1")
    if report.get("kind") == "step" and cells_n1 is not None and cells_n1 < 0.99:
        status = "WARN"; notes.append(f"1-step cell accuracy {cells_n1:.3f} < 0.99")
    if report.get("kind") == "answers" and best_base is not None and cells_last is not None:
        if cells_last < best_base - 0.01 and not name.startswith("only_"):
            status = "WARN"; notes.append(f"last-horizon cells {cells_last:.3f} below baseline {best_base:.3f}")

    return {
        "model": name,
        "kind": report.get("kind"),
        "dataset": report.get("dataset"),
        "epochs": epochs_done,
        "best_epoch": best_epoch,
        "train_seconds": total_s,
        "cells_n1": round(cells_n1, 4) if cells_n1 is not None else None,
        "cells_last": round(cells_last, 4) if cells_last is not None else None,
        "grids_last": round(grids_last, 4) if grids_last is not None else None,
        "delta_vs_best_baseline": round(cells_last - best_base, 4) if (cells_last is not None and best_base is not None) else None,
        "status": status,
        "notes": "; ".join(notes),
    }


def audit_runs(results_dir: Path) -> list[dict]:
    """Audit every model directory containing metrics.json for integrity and baseline deltas."""
    return [audit_run(p.parent) for p in sorted(results_dir.glob("*/metrics.json"))]


def _read_metrics(results_dir: Path, name: str) -> dict | None:
    p = results_dir / name / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def verify_hypotheses(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Check core Deliverable 3 hypotheses and build the multi-task vs single-task table."""
    checks: list[dict] = []

    def add(name: str, passed: bool | None, detail: str):
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        checks.append({"check": name, "status": status, "detail": detail})

    # 1. StepNet exactness on seen, unseen, and 32-cell rows
    for model_name, label in (
        ("step_elementary", "StepNet 16-cell"),
        ("step_elementary_on_elementary_w32_test", "StepNet 32-cell transfer"),
    ):
        rep = _read_metrics(results_dir, model_name)
        if rep is None:
            add(f"{label} exactness (N=64)", None, f"{model_name}/metrics.json not found")
            continue
        for split in ("test", "unseen"):
            if split not in rep["splits"]:
                continue
            last = rep["splits"][split]["all"]["by_horizon"][-1]["model"]
            ok = last["cells_correct"] >= 0.999 and last["grids_correct"] >= 0.99
            add(
                f"{label} exactness ({split}, N={rep['splits'][split]['all']['by_horizon'][-1]['n']})",
                ok,
                f"cells={last['cells_correct']:.4f}, grids={last['grids_correct']:.4f}",
            )

    # 2. AnswerNet beats baselines & confirms Wolfram class separation / chaotic decay
    ans = _read_metrics(results_dir, "answers_elementary")
    if ans is None:
        add("AnswerNet core hypotheses", None, "answers_elementary/metrics.json not found")
    else:
        for split in ("test", "unseen"):
            if split not in ans["splits"]:
                continue
            groups = ans["splits"][split]
            all_last = groups["all"]["by_horizon"][-1]
            m_acc = all_last["model"]["cells_correct"]
            b_acc = max(all_last["copy"]["cells_correct"], all_last["common"]["cells_correct"])
            add(
                f"AnswerNet beats baselines ({split}, N={all_last['n']})",
                m_acc > b_acc,
                f"model={m_acc:.4f} vs best_baseline={b_acc:.4f} (+{m_acc - b_acc:.4f})",
            )
            if {"1 uniform", "2 periodic", "3 chaotic"} <= set(groups):
                c1 = groups["1 uniform"]["by_horizon"][-1]["model"]["cells_correct"]
                c2 = groups["2 periodic"]["by_horizon"][-1]["model"]["cells_correct"]
                c3_first = groups["3 chaotic"]["by_horizon"][0]["model"]["cells_correct"]
                c3_last = groups["3 chaotic"]["by_horizon"][-1]["model"]["cells_correct"]
                add(
                    f"Wolfram class ordering ({split}, N=64): Class 1 & 2 > Class 3",
                    min(c1, c2) > c3_last + 0.05,
                    f"C1={c1:.3f}, C2={c2:.3f}, C3={c3_last:.3f}",
                )
                add(
                    f"Chaotic horizon collapse ({split}): Class 3 drops N=1 -> N=64",
                    c3_first > c3_last + 0.05,
                    f"C3 N=1={c3_first:.3f} -> N=64={c3_last:.3f}",
                )
            cyc = groups["all"].get("cycle_length")
            if cyc:
                add(
                    f"AnswerNet cycle length beats median ({split})",
                    cyc["error"] < cyc["common_error"],
                    f"MAE={cyc['error']:.2f} vs common={cyc['common_error']:.2f}",
                )

    # 3. Multi-task vs single-task comparison
    comparison: list[dict] = []
    if ans is not None:
        task_specs = [
            ("state", "cells_correct (N=64)", lambda s: s["by_horizon"][-1]["model"]["cells_correct"], True),
            ("count", "count_error (N=64)", lambda s: s["by_horizon"][-1]["model"]["count_error"], False),
            ("stable", "stable F1", lambda s: s["stable"]["f1"], True),
            ("cycle", "cycle_length MAE", lambda s: s["cycle_length"]["error"], False),
            ("occurs", "occurs F1", lambda s: s["occurs"]["f1"], True),
        ]
        for split in ("test", "unseen"):
            if split not in ans["splits"]:
                continue
            shared_scores = ans["splits"][split]["all"]
            for task, metric_label, getter, higher_better in task_specs:
                single_rep = _read_metrics(results_dir, f"only_{task}_elementary")
                if single_rep is None or split not in single_rep["splits"]:
                    continue
                sv = float(getter(shared_scores))
                ov = float(getter(single_rep["splits"][split]["all"]))
                delta = sv - ov
                better = "shared" if ((delta > 0) == higher_better) else "single"
                if abs(delta) < 1e-4:
                    better = "tie"
                comparison.append({
                    "split": split,
                    "task": task,
                    "metric": metric_label,
                    "shared": round(sv, 4),
                    "single": round(ov, 4),
                    "delta": round(delta, 4),
                    "winner": better,
                })
    return checks, comparison


def _load_step_checkpoint(checkpoint: Path, dims: int, values: int, cond: int, wrap: bool, device: str) -> StepNet:
    saved = torch.load(checkpoint, map_location=device)
    model = StepNet(dims, values, cond, saved["config"]["width"] or 64, wrap).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


@torch.no_grad()
def _verify_one_step_model(ckpt: Path, datasets_dir: Path, device: str, n_grids: int = 16) -> list[dict]:
    """Cross-check one StepNet checkpoint live against `cellab.engine.step`."""
    model_name = ckpt.parent.name
    if model_name == "step_elementary":
        elem_spec_path = datasets_dir / "elementary_w16.json"
        if not elem_spec_path.exists():
            return []
        spec, _ = load_spec(elem_spec_path)
        base_auto = load_automaton((elem_spec_path.parent / spec.automaton).resolve())
        model = _load_step_checkpoint(ckpt, dims=1, values=2, cond=8, wrap=True, device=device)
        test_rules = [0, 30, 90, 110, 184]
        out = []
        for width in (16, 32):
            total_cells, match_cells, total_grids, match_grids = 0, 0, 0, 0
            rng = make_generator(99000 + width)
            for rule in test_rules:
                auto = dataclasses.replace(
                    base_auto,
                    parameters={**base_auto.parameters, "rule": rule},
                    constants={**base_auto.constants, "rule": rule},
                )
                inits = rng.integers(0, 2, size=(n_grids, width), dtype=np.uint8)
                truth = np.empty_like(inits)
                for i in range(n_grids):
                    g = inits[i]
                    for t in range(64):
                        g = engine_step(auto, g, rng, t)
                    truth[i] = g
                g_t = torch.as_tensor(inits, dtype=torch.long, device=device)
                c_t = torch.as_tensor(eca.rule_bits(rule), dtype=torch.float32, device=device).unsqueeze(0).expand(n_grids, -1)
                pred = model.rollout(g_t, c_t, 64).cpu().numpy()
                eq = pred == truth
                match_cells += int(eq.sum())
                total_cells += int(eq.size)
                match_grids += int(eq.all(axis=1).sum())
                total_grids += n_grids
            out.append({
                "model": "step_elementary",
                "target": f"live engine rules {test_rules}, width={width}, N=64",
                "cells_match": round(match_cells / total_cells, 5),
                "grids_match": round(match_grids / total_grids, 5),
                "status": "PASS" if match_grids == total_grids else "FAIL",
            })
        return out

    ds_name = model_name[len("step_"):]
    spec_path = datasets_dir / f"{ds_name}.json"
    if not spec_path.exists():
        return []
    spec, _ = load_spec(spec_path)
    auto = load_automaton((spec_path.parent / spec.automaton).resolve())
    if not auto.deterministic or auto.layers or auto.states.kind != "int" or auto.states.highest is None:
        return []
    lowest = int(auto.states.lowest)
    values = int(auto.states.highest) - lowest + 1
    model = _load_step_checkpoint(ckpt, dims=auto.dimensions, values=values, cond=0,
                                  wrap=auto.edge_mode == "wrap", device=device)
    rng = make_generator(77000)
    inits = rng.integers(lowest, lowest + values, size=(n_grids, *spec.size), dtype=auto.states.dtype)
    truth_1 = np.empty_like(inits)
    truth_16 = np.empty_like(inits)
    for i in range(n_grids):
        g = inits[i]
        for t in range(16):
            g = engine_step(auto, g, rng, t)
            if t == 0:
                truth_1[i] = g
        truth_16[i] = g
    g_t = torch.as_tensor(inits.astype(np.int64) - lowest, dtype=torch.long, device=device)
    c_t = torch.zeros((n_grids, 0), dtype=torch.float32, device=device)
    pred_1 = model.rollout(g_t, c_t, 1).cpu().numpy() + lowest
    pred_16 = model.rollout(g_t, c_t, 16).cpu().numpy() + lowest
    acc_1 = float((pred_1 == truth_1).mean())
    acc_16 = float((pred_16 == truth_16).mean())
    return [{
        "model": model_name,
        "target": f"live engine {ds_name}, N=1 & N=16",
        "cells_match": round(acc_1, 5),
        "cells_match_n16": round(acc_16, 5),
        "status": "PASS" if acc_1 >= 0.99 else "WARN",
    }]


@torch.no_grad()
def verify_against_simulator(results_dir: Path, datasets_dir: Path, device: str, n_grids: int = 16) -> list[dict]:
    """Cross-check all trained StepNet models directly against live `cellab.engine.step` rollouts."""
    results: list[dict] = []
    elem_ckpt = results_dir / "step_elementary" / "best.pt"
    if elem_ckpt.exists():
        results.extend(_verify_one_step_model(elem_ckpt, datasets_dir, device, n_grids))
    for ckpt in sorted(results_dir.glob("step_*/best.pt")):
        if ckpt.parent.name != "step_elementary":
            results.extend(_verify_one_step_model(ckpt, datasets_dir, device, n_grids))
    return results


def validate_model(results_dir: str | Path, name: str, datasets_dir: str | Path = "datasets",
                   device: str | None = None) -> dict:
    """Fast per-model post-training validation: audit metrics + live engine check for StepNets."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    folder = Path(results_dir) / name
    row = audit_run(folder)
    sim_checks: list[dict] = []
    if row["kind"] == "step" and (folder / "best.pt").exists():
        sim_checks = _verify_one_step_model(folder / "best.pt", Path(datasets_dir), device, n_grids=8)
    statuses = [row["status"], *(c["status"] for c in sim_checks)]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    parts = [f"N=1:{row['cells_n1']}", f"N=last:{row['cells_last']}"]
    if row["delta_vs_best_baseline"] is not None:
        parts.append(f"vs_base:{row['delta_vs_best_baseline']:+.4f}")
    if sim_checks:
        parts.append("live_sim:" + ",".join(f"{c['cells_match']:.4f}" for c in sim_checks))
    if row["notes"]:
        parts.append(row["notes"])
    print(f"[validate:{overall}] {name} | " + " | ".join(parts), flush=True)
    return {"status": overall, "audit": row, "live_sim": sim_checks}


@torch.no_grad()
def benchmark_inference(results_dir: Path, datasets_dir: Path, device: str) -> list[dict]:
    """Benchmark wall-clock latency & throughput: live simulator vs StepNet rollout vs AnswerNet."""
    rows: list[dict] = []
    elem_spec_path = datasets_dir / "elementary_w16.json"

    # 1. Exact simulator baseline (CPU, single sample)
    if elem_spec_path.exists():
        spec, _ = load_spec(elem_spec_path)
        base_auto = load_automaton((elem_spec_path.parent / spec.automaton).resolve())
        auto = dataclasses.replace(
            base_auto,
            parameters={**base_auto.parameters, "rule": 30},
            constants={**base_auto.constants, "rule": 30},
        )
        rng = make_generator(42)
        init = rng.integers(0, 2, size=(16,), dtype=np.uint8)
        for n in (1, 8, 64):
            reps = 60 if n <= 8 else 25
            t0 = time.perf_counter()
            for _ in range(reps):
                g = init
                for t in range(n):
                    g = engine_step(auto, g, rng, t)
            dt = (time.perf_counter() - t0) / reps
            rows.append({
                "method": "cellab simulator (CPU)",
                "horizon_n": n,
                "batch": 1,
                "ms_per_sample": round(dt * 1000, 4),
                "samples_per_sec": round(1.0 / max(dt, 1e-9), 1),
            })

    # 2. StepNet rollout
    step_ckpt = results_dir / "step_elementary" / "best.pt"
    if step_ckpt.exists():
        model = _load_step_checkpoint(step_ckpt, dims=1, values=2, cond=8, wrap=True, device=device)
        for batch in (1, 256):
            grid = torch.randint(0, 2, (batch, 16), device=device)
            cond = torch.rand((batch, 8), device=device)
            for n in (1, 8, 64):
                _ = model.rollout(grid, cond, n)
                _sync(device)
                reps = 20 if n <= 8 else 8
                t0 = time.perf_counter()
                for _ in range(reps):
                    _ = model.rollout(grid, cond, n)
                _sync(device)
                dt = (time.perf_counter() - t0) / (reps * batch)
                rows.append({
                    "method": f"StepNet rollout ({device})",
                    "horizon_n": n,
                    "batch": batch,
                    "ms_per_sample": round(dt * 1000, 5),
                    "samples_per_sec": round(1.0 / max(dt, 1e-9), 1),
                })

    # 3. AnswerNet single-pass (all 5 questions at once)
    ans_ckpt = results_dir / "answers_elementary" / "best.pt"
    if ans_ckpt.exists():
        saved = torch.load(ans_ckpt, map_location=device)
        model = AnswerNet(dims=1, values=2, cond=8, extra=0, width=saved["config"]["width"] or 96, wrap=True).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        for batch in (1, 256):
            grid = torch.randint(0, 2, (batch, 16), device=device)
            extra = torch.zeros((batch, 0, 16), device=device)
            query = torch.randint(0, 2, (batch, 16), device=device)
            cond = torch.rand((batch, 8), device=device)
            n_t = torch.full((batch,), 64, device=device)
            _ = model(grid, extra, query, cond, n_t)
            _sync(device)
            reps = 30
            t0 = time.perf_counter()
            for _ in range(reps):
                _ = model(grid, extra, query, cond, n_t)
            _sync(device)
            dt = (time.perf_counter() - t0) / (reps * batch)
            rows.append({
                "method": f"AnswerNet 1-pass ({device})",
                "horizon_n": 64,
                "batch": batch,
                "ms_per_sample": round(dt * 1000, 5),
                "samples_per_sec": round(1.0 / max(dt, 1e-9), 1),
            })

    # 4. 2D benchmark (16x16 Game of Life or Brian's Brain if trained)
    for ds_2d in ("game_of_life_16", "brians_brain_16"):
        step_2d = results_dir / f"step_{ds_2d}" / "best.pt"
        ans_2d = results_dir / f"answers_{ds_2d}" / "best.pt"
        spec_2d = datasets_dir / f"{ds_2d}.json"
        if not (spec_2d.exists() and (step_2d.exists() or ans_2d.exists())):
            continue
        spec, _ = load_spec(spec_2d)
        auto = load_automaton((spec_2d.parent / spec.automaton).resolve())
        values = int(auto.states.highest) - int(auto.states.lowest) + 1
        rng = make_generator(43)
        init = rng.integers(int(auto.states.lowest), int(auto.states.highest) + 1, size=tuple(spec.size), dtype=auto.states.dtype)
        reps = 15
        t0 = time.perf_counter()
        for _ in range(reps):
            g = init
            for t in range(32):
                g = engine_step(auto, g, rng, t)
        dt = (time.perf_counter() - t0) / reps
        rows.append({
            "method": f"cellab 2D {ds_2d} (CPU)",
            "horizon_n": 32,
            "batch": 1,
            "ms_per_sample": round(dt * 1000, 4),
            "samples_per_sec": round(1.0 / max(dt, 1e-9), 1),
        })
        if step_2d.exists():
            m = _load_step_checkpoint(step_2d, dims=2, values=values, cond=0, wrap=auto.edge_mode == "wrap", device=device)
            grid = torch.randint(0, values, (64, *spec.size), device=device)
            cond = torch.zeros((64, 0), device=device)
            _ = m.rollout(grid, cond, 32); _sync(device)
            t0 = time.perf_counter()
            for _ in range(5):
                _ = m.rollout(grid, cond, 32)
            _sync(device)
            dt = (time.perf_counter() - t0) / (5 * 64)
            rows.append({
                "method": f"StepNet 2D {ds_2d} ({device})",
                "horizon_n": 32,
                "batch": 64,
                "ms_per_sample": round(dt * 1000, 5),
                "samples_per_sec": round(1.0 / max(dt, 1e-9), 1),
            })
        if ans_2d.exists():
            saved = torch.load(ans_2d, map_location=device)
            m = AnswerNet(dims=2, values=values, cond=0, extra=0, width=saved["config"]["width"] or 96,
                          wrap=auto.edge_mode == "wrap").to(device)
            m.load_state_dict(saved["model"]); m.eval()
            grid = torch.randint(0, values, (64, *spec.size), device=device)
            extra = torch.zeros((64, 0, *spec.size), device=device)
            cond = torch.zeros((64, 0), device=device)
            n_t = torch.full((64,), 32, device=device)
            _ = m(grid, extra, grid, cond, n_t); _sync(device)
            t0 = time.perf_counter()
            for _ in range(15):
                _ = m(grid, extra, grid, cond, n_t)
            _sync(device)
            dt = (time.perf_counter() - t0) / (15 * 64)
            rows.append({
                "method": f"AnswerNet 2D {ds_2d} ({device})",
                "horizon_n": 32,
                "batch": 64,
                "ms_per_sample": round(dt * 1000, 5),
                "samples_per_sec": round(1.0 / max(dt, 1e-9), 1),
            })
        break

    return rows


def plot_verification(report: dict, out_dir: Path) -> list[str]:
    """Generate presentation-ready verification plots (multitask vs single, speed benchmark)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    created: list[str] = []

    # 1. Multi-task vs single-task (higher-is-better normalized tasks: state, stable, occurs)
    comp = [r for r in report.get("multitask_vs_single", []) if r["split"] == "test"]
    if comp:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        acc_rows = [r for r in comp if r["task"] in ("state", "stable", "occurs")]
        err_rows = [r for r in comp if r["task"] in ("count", "cycle")]

        if acc_rows:
            x = np.arange(len(acc_rows))
            axes[0].bar(x - 0.18, [r["shared"] for r in acc_rows], 0.35, label="Shared (5 heads)")
            axes[0].bar(x + 0.18, [r["single"] for r in acc_rows], 0.35, label="Separate (1 head)")
            axes[0].set_xticks(x, [f"{r['task']}\n({r['metric']})" for r in acc_rows], fontsize=8)
            axes[0].set_ylim(0, 1.05)
            axes[0].set_title("Accuracy / F1 (higher is better)")
            axes[0].legend(fontsize=8)

        if err_rows:
            x = np.arange(len(err_rows))
            axes[1].bar(x - 0.18, [r["shared"] for r in err_rows], 0.35, label="Shared (5 heads)")
            axes[1].bar(x + 0.18, [r["single"] for r in err_rows], 0.35, label="Separate (1 head)")
            axes[1].set_xticks(x, [f"{r['task']}\n({r['metric']})" for r in err_rows], fontsize=8)
            axes[1].set_title("Error / MAE (lower is better)")
            axes[1].legend(fontsize=8)

        fig.suptitle("One Shared Network vs Five Separate Networks (test split)")
        fig.tight_layout()
        p = out_dir / "multitask_vs_single.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p))

    # 2. Speed benchmark at N=64
    bench = [r for r in report.get("speed_benchmarks", []) if r["horizon_n"] == 64]
    if bench:
        labels = [f"{r['method']}\n(batch={r['batch']})" for r in bench]
        vals = [r["samples_per_sec"] for r in bench]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, vals)
        ax.set_xscale("log")
        ax.set_xlabel("Throughput (samples / second, log scale)")
        ax.set_title("Inference Throughput at N = 64 steps (Elementary width 16)")
        for i, v in enumerate(vals):
            ax.text(v * 1.08, i, f"{v:,.0f}/s", va="center", fontsize=8)
        fig.tight_layout()
        p = out_dir / "speed_benchmark.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p))

    return created


def format_markdown_report(report: dict) -> str:
    """Format the verification and benchmark report as Markdown for Drive/docs."""
    s = report["summary"]
    lines = [
        f"# Verification & Benchmark Report (`{s['device']}`)",
        "",
        f"- **Models audited:** {s['models_audited']}",
        f"- **PASS:** {s['passed']} | **WARN:** {s['warnings']} | **FAIL:** {s['failed']} | **SKIP:** {s['skipped']}",
        "",
    ]
    if report["hypotheses"]:
        lines += ["## Core Hypothesis Checks", "", "| Status | Check | Detail |", "|---|---|---|"]
        for c in report["hypotheses"]:
            lines.append(f"| {c['status']} | {c['check']} | {c['detail']} |")
        lines.append("")
    if report["live_simulator_checks"]:
        lines += ["## Live Simulator Cross-Checks", "", "| Status | Model | Target | Cells match | Extra |", "|---|---|---|---|---|"]
        for r in report["live_simulator_checks"]:
            extra = f"grids={r['grids_match']:.4f}" if "grids_match" in r else (f"N=16={r['cells_match_n16']:.4f}" if "cells_match_n16" in r else "")
            lines.append(f"| {r['status']} | `{r['model']}` | {r['target']} | {r['cells_match']:.4f} | {extra} |")
        lines.append("")
    if report["multitask_vs_single"]:
        lines += ["## Shared (5-head) vs Separate (1-head)", "", "| Split | Task | Metric | Shared | Single | Delta | Winner |", "|---|---|---|---|---|---|---|"]
        for r in report["multitask_vs_single"]:
            lines.append(f"| {r['split']} | `{r['task']}` | {r['metric']} | {r['shared']:.4f} | {r['single']:.4f} | {r['delta']:+.4f} | {r['winner']} |")
        lines.append("")
    if report["speed_benchmarks"]:
        lines += ["## Inference Speed Benchmarks", "", "| Method | Horizon N | Batch | ms / sample | Samples / s |", "|---|---|---|---|---|"]
        for r in report["speed_benchmarks"]:
            lines.append(f"| {r['method']} | {r['horizon_n']} | {r['batch']} | {r['ms_per_sample']:.4f} | {r['samples_per_sec']:,.1f} |")
        lines.append("")
    if report["audit"]:
        lines += ["## Model Audit", "", "| Status | Model | Epochs | Best ep | Time (s) | Cells N=1 | Cells last | vs Base | Notes |", "|---|---|---|---|---|---|---|---|---|"]
        for r in report["audit"]:
            delta = f"{r['delta_vs_best_baseline']:+.4f}" if r["delta_vs_best_baseline"] is not None else "—"
            lines.append(f"| {r['status']} | `{r['model']}` | {r['epochs']} | {r['best_epoch']} | {r['train_seconds']} | {r['cells_n1']} | {r['cells_last']} | {delta} | {r['notes']} |")
        lines.append("")
    return "\n".join(lines)


def run_verification(results_dir: str | Path, datasets_dir: str | Path = "datasets",
                     device: str | None = None, save: bool = True) -> dict:
    """Run all post-training verification checks and speed benchmarks."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res_path = Path(results_dir)
    ds_path = Path(datasets_dir)

    audit = audit_runs(res_path)
    hypotheses, comparison = verify_hypotheses(res_path)
    live_sim = verify_against_simulator(res_path, ds_path, device)
    benchmarks = benchmark_inference(res_path, ds_path, device)

    all_statuses = [r["status"] for r in audit + hypotheses + live_sim]
    summary = {
        "device": device,
        "models_audited": len(audit),
        "passed": all_statuses.count("PASS"),
        "warnings": all_statuses.count("WARN"),
        "failed": all_statuses.count("FAIL"),
        "skipped": all_statuses.count("SKIP"),
    }
    report = {
        "summary": summary,
        "hypotheses": hypotheses,
        "multitask_vs_single": comparison,
        "live_simulator_checks": live_sim,
        "speed_benchmarks": benchmarks,
        "audit": audit,
    }
    if save:
        res_path.mkdir(parents=True, exist_ok=True)
        (res_path / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (res_path / "verification_report.md").write_text(format_markdown_report(report), encoding="utf-8")
        report["plots"] = plot_verification(report, res_path)
    return report


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"=== Verification Summary ({s['device']}) ===")
    print(f"Models audited: {s['models_audited']} | PASS: {s['passed']} | WARN: {s['warnings']} | FAIL: {s['failed']} | SKIP: {s['skipped']}")
    if report["hypotheses"]:
        print("\n--- Core Hypothesis Checks ---")
        for c in report["hypotheses"]:
            print(f"[{c['status']:4s}] {c['check']}: {c['detail']}")
    if report["live_simulator_checks"]:
        print("\n--- Live Simulator Cross-Checks ---")
        for r in report["live_simulator_checks"]:
            extra = f", N=16={r['cells_match_n16']:.4f}" if "cells_match_n16" in r else ""
            print(f"[{r['status']:4s}] {r['model']} ({r['target']}): cells={r['cells_match']:.4f}{extra}")
    if report["multitask_vs_single"]:
        print("\n--- Shared (5-head) vs Separate (1-head) ---")
        for r in report["multitask_vs_single"]:
            print(f"  {r['split']:6s} | {r['task']:6s} ({r['metric']:20s}): shared={r['shared']:.4f}  single={r['single']:.4f}  delta={r['delta']:+.4f} ({r['winner']})")
    if report["speed_benchmarks"]:
        print("\n--- Inference Speed Benchmarks ---")
        for r in report["speed_benchmarks"]:
            print(f"  {r['method']:28s} N={r['horizon_n']:2d} batch={r['batch']:3d} -> {r['ms_per_sample']:.4f} ms/sample ({r['samples_per_sec']:,.1f} samples/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", help="folder containing trained model subfolders")
    parser.add_argument("--datasets", default="datasets", help="folder containing dataset spec JSON files")
    parser.add_argument("--device")
    args = parser.parse_args()
    report = run_verification(args.results, args.datasets, args.device)
    print_report(report)


if __name__ == "__main__":
    main()
