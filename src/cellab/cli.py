"""The command line. Every command only calls the library; no logic lives here.

    cellab check  FILE...               check automaton and experiment files
    cellab run    EXPERIMENT [options]  start a run, answer its questions
    cellab resume RUN_FOLDER [options]  continue a stopped (or finished) run
    cellab answers RUN_FOLDER           print a finished run's answers
    cellab show   RUN_FOLDER [--step K] print one stored generation
    cellab render RUN_FOLDER            make PNG/GIF pictures of a run
    cellab dataset DATASET_FILE         generate a machine-learning dataset
    cellab qa     DATASET.npz           write a dataset as questions and answers in words
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .automaton import load_automaton
from .dataset import build_dataset, write_question_answer_pairs
from .errors import CellabError
from .experiment import load_experiment
from .render import render_run
from .runner import Run, RunResult, RunState
from .storage import RunFolder


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except CellabError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cellab", description="Simulate cellular automata described in files.")
    parser.add_argument("--version", action="version", version=f"cellab {__version__}")
    commands = parser.add_subparsers(dest="command")

    check = commands.add_parser("check", help="check automaton and experiment files")
    check.add_argument("files", nargs="+", type=Path)
    check.set_defaults(handler=_check)

    run = commands.add_parser("run", help="start a run of an experiment")
    run.add_argument("experiment", type=Path)
    run.add_argument("--seed", type=int, help="use this seed instead of the experiment's")
    run.add_argument("--steps", type=int, help="number of steps instead of the experiment's")
    run.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where run folders are made (default: runs)")
    run.add_argument("--out", type=Path, help="exact run folder to create")
    _add_progress_options(run)
    run.set_defaults(handler=_run)

    resume = commands.add_parser("resume", help="continue a run from its last checkpoint")
    resume.add_argument("folder", type=Path)
    resume.add_argument("--steps", type=int, help="run this many more steps beyond the current one")
    _add_progress_options(resume)
    resume.set_defaults(handler=_resume)

    answers = commands.add_parser("answers", help="print a finished run's answers")
    answers.add_argument("folder", type=Path)
    answers.set_defaults(handler=_answers)

    show = commands.add_parser("show", help="print one stored generation of a run")
    show.add_argument("folder", type=Path)
    show.add_argument("--step", type=int, help="which generation (default: the last stored)")
    show.set_defaults(handler=_show)

    render = commands.add_parser("render", help="make PNG/GIF pictures of a run")
    render.add_argument("folder", type=Path)
    render.add_argument("--out", type=Path, help="folder for the pictures (default: the run folder)")
    render.add_argument("--until", type=int, help="only draw generations 0..UNTIL")
    render.set_defaults(handler=_render)

    dataset = commands.add_parser("dataset", help="generate a machine-learning dataset from a dataset file")
    dataset.add_argument("spec", type=Path)
    dataset.add_argument("--out", type=Path, default=Path("data"), help="folder for the .npz and .json (default: data)")
    dataset.add_argument("--workers", type=int, default=1, help="processes to use (default: 1)")
    dataset.set_defaults(handler=_dataset)

    qa = commands.add_parser("qa", help="write a dataset as questions and answers in words (JSON lines)")
    qa.add_argument("dataset", type=Path)
    qa.add_argument("--out", type=Path, help="output file; end it in .gz to compress (default: next to the dataset)")
    qa.add_argument("--limit", type=int, help="only the first LIMIT samples")
    qa.add_argument("--every", type=int, default=1, help="only every EVERY-th sample (default: all)")
    qa.set_defaults(handler=_qa)
    return parser


def _add_progress_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--stop-at", type=int, help="stop cleanly at this step (as if Ctrl+C was pressed)")
    command.add_argument("--trace", action="store_true", help="print every step's fingerprint")
    command.add_argument("--show", action="store_true", help="print the grid after every step")


# ------------------------------------------------------------ commands

def _check(args) -> int:
    failures = 0
    for path in args.files:
        try:
            if _looks_like_experiment(path):
                experiment = load_experiment(path)
                print(f"OK  {path}: experiment '{experiment.name}' "
                      f"({experiment.automaton.name}, grid {'x'.join(map(str, experiment.shape))}, "
                      f"{len(experiment.questions)} question(s))")
            else:
                automaton = load_automaton(path)
                kind = "deterministic" if automaton.deterministic else "uses random numbers"
                print(f"OK  {path}: automaton '{automaton.name}' ({len(automaton.rules)} rule(s), {kind})")
        except CellabError as error:
            failures += 1
            print(f"ERR {error}")
    return 1 if failures else 0


def _run(args) -> int:
    experiment = load_experiment(args.experiment)
    if args.steps is not None:
        experiment = experiment.with_steps(args.steps)
    run = Run.start(experiment, runs_dir=args.runs_dir, seed=args.seed, folder=args.out)
    print(f"Run folder: {run.folder.path}")
    print(f"Automaton:  {experiment.automaton.name}   seed: {run.state.seed}")
    _progress(run, args)(run.state)
    return _finish(run.advance(stop_at=args.stop_at, on_step=_progress(run, args)))


def _resume(args) -> int:
    run = Run.open(args.folder)
    if args.steps is not None:
        run.extend(args.steps)
    print(f"Resuming {run.folder.path} at step {run.state.step}")
    return _finish(run.advance(stop_at=args.stop_at, on_step=_progress(run, args)))


def _answers(args) -> int:
    for number, answer in enumerate(RunFolder(args.folder).read_answers(), start=1):
        _print_answer(number, answer["question"], answer["answer"])
    return 0


def _show(args) -> int:
    folder = RunFolder(args.folder)
    experiment = load_experiment(folder.file(RunFolder.EXPERIMENT))
    dtype = experiment.automaton.state_dtype
    shape = experiment.automaton.state_shape(experiment.shape)
    count = folder.records(RunFolder.GENERATIONS, int(_cells(shape)) * dtype.itemsize)
    if count == 0:
        raise CellabError(f"{folder.path} has no stored generations")
    step = count - 1 if args.step is None else args.step
    grid = folder.read_generation(step, shape, dtype)
    print(f"Step {step}:")
    print("\n".join(experiment.format(grid)))
    return 0


def _render(args) -> int:
    for path in render_run(args.folder, args.out, until=args.until):
        print(f"wrote {path}")
    return 0


def _dataset(args) -> int:
    def progress(done: int, total: int) -> None:
        print(f"{done}/{total} settings done", flush=True)
    path = build_dataset(args.spec, args.out, args.workers, progress)
    print(f"wrote {path} and {path.with_suffix('.json')}")
    return 0


def _qa(args) -> int:
    out = args.out or args.dataset.with_name(args.dataset.stem + "_qa.jsonl.gz")
    print(f"wrote {write_question_answer_pairs(args.dataset, out, args.limit, args.every):,} questions and answers to {out}")
    return 0


# ------------------------------------------------------------ helpers

def _progress(run: Run, args):
    def on_step(state: RunState) -> None:
        if args.trace:
            print(f"step {state.step:>6}  fingerprint {state.history.fingerprints[-1].hex()[:16]}")
        if args.show:
            print(f"-- step {state.step}")
            print("\n".join(run.experiment.format(state.grid)))
    return on_step


def _finish(result: RunResult) -> int:
    folder = result.run.folder.path
    if result.stopped:
        print(f"Stopped at step {result.run.state.step}. Continue with:  cellab resume {folder}")
        return 0
    print(f"Finished at step {result.run.state.step}.")
    for number, answer in enumerate(result.answers, start=1):
        _print_answer(number, answer.question, answer.text)
    print(f"Saved in {folder}")
    return 0


def _print_answer(number: int, question: dict, text: str) -> None:
    details = ", ".join(f"{k}={v}" for k, v in question.items() if k not in ("ask", "rows"))
    print(f"\n[{number}] {question['ask']}" + (f" ({details})" if details else ""))
    for line in text.splitlines():
        print(f"    {line}")


def _looks_like_experiment(path: Path) -> bool:
    try:
        return "automaton" in json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False


def _cells(shape) -> int:
    total = 1
    for n in shape:
        total *= n
    return total
