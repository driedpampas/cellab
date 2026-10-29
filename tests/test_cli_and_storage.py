"""The command line, the run folder layout, and the pictures."""

import json

import numpy as np
from PIL import Image

from cellab import Run
from cellab.cli import main
from cellab.render import render_run
from cellab.storage import RunFolder
from helpers import AUTOMATA, EXPERIMENTS, FOREST_FIRE, example_experiment, run_experiment, write_json


def test_check_accepts_good_files_and_reports_bad_ones(tmp_path, capsys):
    bad = write_json(tmp_path / "bad.json", dict(FOREST_FIRE, rules=[{"when": "self == FIER", "then": "FIRE"}]))
    code = main(["check", str(AUTOMATA / "forest_fire.json"), str(EXPERIMENTS / "traffic_brief.json"), str(bad)])
    out = capsys.readouterr().out
    assert code == 1
    assert "OK  " in out and "automaton 'Forest fire'" in out and "experiment 'Traffic" in out
    assert "ERR" in out and "did you mean 'FIRE'?" in out


def test_run_prints_answers_and_fills_the_run_folder(tmp_path, capsys):
    folder = tmp_path / "run"
    assert main(["run", str(EXPERIMENTS / "forest_fire_brief.json"), "--out", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "Stable from step 5" in out
    for name in (RunFolder.AUTOMATON, RunFolder.EXPERIMENT, RunFolder.CHECKPOINT, RunFolder.FINGERPRINTS,
                 RunFolder.GENERATIONS, RunFolder.ANSWERS, RunFolder.STATUS):
        assert (folder / name).is_file(), name
    copied = json.loads((folder / RunFolder.EXPERIMENT).read_text())
    assert copied["automaton"] == "automaton.json" and isinstance(copied["seed"], int)


def test_stop_resume_answers_and_show_from_the_command_line(tmp_path, capsys):
    folder = tmp_path / "run"
    main(["run", str(EXPERIMENTS / "traffic_brief.json"), "--out", str(folder), "--stop-at", "4"])
    assert "Stopped at step 4" in capsys.readouterr().out
    assert main(["resume", str(folder)]) == 0
    assert "Resuming" in capsys.readouterr().out
    assert main(["answers", str(folder)]) == 0
    assert "4 cells with value CAR" in capsys.readouterr().out
    assert main(["show", str(folder), "--step", "1"]) == 0
    assert "...CC.C..C" in capsys.readouterr().out


def test_trace_prints_one_fingerprint_per_step(tmp_path, capsys):
    main(["run", str(EXPERIMENTS / "rule58_lab.json"), "--out", str(tmp_path / "r"), "--trace"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("step")]
    assert len(lines) == 11                                   # steps 0..10


def test_errors_are_short_messages_not_tracebacks(tmp_path, capsys):
    assert main(["run", str(tmp_path / "missing.json")]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["resume", str(tmp_path)]) == 1
    assert "not a run folder" in capsys.readouterr().err


def test_generations_on_disk_match_the_run(tmp_path):
    run, _ = run_experiment(example_experiment("brians_brain_brief"), tmp_path)
    shape, dtype = run.state.grid.shape, run.state.grid.dtype
    stored = run.folder.read_all_generations(shape, dtype)
    assert len(stored) == run.state.step + 1
    assert (stored[-1] == run.state.grid).all()
    assert run.folder.records(RunFolder.FINGERPRINTS, 16) == run.state.step + 1


def test_replay_without_stored_generations_gives_the_same_grids(tmp_path):
    experiment = example_experiment("forest_fire_brief")
    stored_run, _ = run_experiment(experiment, tmp_path / "a")
    unstored = experiment.with_steps(experiment.steps)
    import dataclasses
    unstored = dataclasses.replace(unstored, store_generations=False)
    replay_run, _ = run_experiment(unstored, tmp_path / "b")
    for k in range(experiment.steps + 1):
        assert (stored_run.grid_at(k) == replay_run.grid_at(k)).all()


def test_render_makes_pictures(tmp_path):
    run_2d, _ = run_experiment(example_experiment("forest_fire_brief"), tmp_path)
    files = {path.name for path in render_run(run_2d.folder.path)}
    assert files == {"first.png", "last.png", "run.gif"}
    gif = Image.open(run_2d.folder.file("run.gif"))
    total = 0
    for frame in range(gif.n_frames):          # identical frames (a stable grid) are merged by Pillow,
        gif.seek(frame)                         # so check the total time instead of the frame count
        total += gif.info["duration"]
    assert total == (run_2d.state.step + 1) * 120

    run_1d, _ = run_experiment(example_experiment("rule30_triangle"), tmp_path / "1d")
    picture = Image.open(render_run(run_1d.folder.path)[0])
    cells = np.array(picture.convert("L"))
    assert picture.height // picture.width == (run_1d.state.step + 1) // 63   # one row per generation
    assert cells.min() < 100 < cells.max()                                   # both colours are present


def test_render_can_stop_at_a_given_step(tmp_path, capsys):
    run, _ = run_experiment(example_experiment("rule30_triangle"), tmp_path)
    assert run.state.step > 31                       # the questions kept the run going
    assert main(["render", str(run.folder.path), "--out", str(tmp_path / "pics"), "--until", "31"]) == 0
    picture = Image.open(tmp_path / "pics" / "spacetime.png")
    assert picture.height * 63 == picture.width * 32   # 32 generations of 63 cells
