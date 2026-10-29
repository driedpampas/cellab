r"""Running an experiment: the loop that connects every other part.

    Run.start(experiment)                    Run.open(run folder)   (to resume)
        seed -> generator                        read the folder's own copies of the files
        generation 0 (initial world)             check automaton.json was not edited
        new run folder, first checkpoint         restore grid, step, generator, fingerprints
                       \                        /
                        run.advance()  -- the loop -----------------------------------
                        while the run should continue:
                            stop requested (Ctrl+C or stop_at)?  -> checkpoint, "stopped"
                            grid = engine.step(grid)
                            fingerprint it, append it, keep it if a question needs it
                            every `checkpoint_every` steps: checkpoint
                        -> answers.json, "finished"

The run continues while the experiment's `steps` are not done, or while some
question is still unresolved and `max_steps` is not reached (see questions.py).

Ctrl+C does not interrupt a step half-way: it only sets a flag, and the loop stops
cleanly between two steps after saving a checkpoint. Resuming from that checkpoint
gives exactly the same results as a run that was never stopped.
"""

import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .engine import step
from .errors import CellabError, StorageError
from .experiment import Experiment, load_experiment, make_initial_grid
from .fingerprints import FINGERPRINT_SIZE, History, fingerprint
from .questions import Answer, AnswerContext
from .randomness import generator_state, make_generator, new_seed, restore_generator
from .storage import RecordWriter, RunFolder


@dataclass
class RunState:
    """Everything that changes while a run goes on. A checkpoint saves exactly this."""
    step: int
    grid: np.ndarray
    initial: np.ndarray
    seed: int
    rng: np.random.Generator
    history: History
    captured: dict[int, np.ndarray]          # grids kept for questions, by step

    def rng_state(self) -> str:
        return generator_state(self.rng)


@dataclass
class RunResult:
    run: "Run"
    stopped: bool                            # True if stopped early (Ctrl+C or stop_at)
    answers: list[Answer] | None             # None while the run is not finished


class Run:
    def __init__(self, experiment: Experiment, folder: RunFolder, state: RunState):
        self.experiment = experiment
        self.folder = folder
        self.state = state

    # ------------------------------------------------------------ starting and reopening
    @classmethod
    def start(cls, experiment: Experiment, runs_dir: str | Path = "runs", seed: int | None = None,
              folder: str | Path | None = None) -> "Run":
        """Create generation 0 and a new run folder. Call advance() to run it."""
        if seed is None:
            seed = experiment.seed if experiment.seed is not None else new_seed()
        rng = make_generator(seed)
        grid = make_initial_grid(experiment, rng)
        run_folder = RunFolder.create(experiment, seed, runs_dir, folder)
        state = RunState(step=0, grid=grid, initial=grid.copy(), seed=seed, rng=rng,
                         history=History(), captured={})
        run = cls(experiment, run_folder, state)
        with run._writers() as writers:
            run._record(writers)
            run._checkpoint(writers)
        return run

    @classmethod
    def open(cls, path: str | Path) -> "Run":
        """Reopen a run folder at its last checkpoint, ready to continue."""
        folder = RunFolder(path)
        if not folder.file(RunFolder.EXPERIMENT).is_file():
            raise StorageError(f"{folder.path} is not a run folder (no {RunFolder.EXPERIMENT})")
        experiment = load_experiment(folder.file(RunFolder.EXPERIMENT))
        saved = folder.load_checkpoint()
        if experiment.automaton.digest != saved["automaton_digest"]:
            raise StorageError(f"{folder.file(RunFolder.AUTOMATON)} was changed after the run started; "
                               "continuing would mix two different automata. Start a new run instead.")
        count = saved["step"] + 1                 # steps 0..step are saved
        history = History.rebuild(folder.read_fingerprints(count))
        folder.truncate(RunFolder.FINGERPRINTS, FINGERPRINT_SIZE, count)
        if experiment.store_generations:
            folder.truncate(RunFolder.GENERATIONS, saved["grid"].nbytes, count)
        state = RunState(step=saved["step"], grid=saved["grid"], initial=saved["initial"], seed=saved["seed"],
                         rng=restore_generator(saved["rng_state"]), history=history, captured=saved["captured"])
        return cls(experiment, folder, state)

    def extend(self, more_steps: int) -> None:
        """Run `more_steps` beyond the current step (e.g. to continue a finished run)."""
        self.experiment = self.experiment.with_steps(self.state.step + more_steps)
        self.folder.update_experiment(self.experiment)

    # ------------------------------------------------------------ the loop
    def advance(self, stop_at: int | None = None,
                on_step: Callable[[RunState], None] | None = None) -> RunResult:
        """Run until done, or until step `stop_at`, or until Ctrl+C."""
        automaton = self.experiment.automaton
        try:
            with self._writers() as writers, _stop_requests() as stop:
                while self._should_continue():
                    if stop.requested or (stop_at is not None and self.state.step >= stop_at):
                        self._checkpoint(writers)
                        self.folder.write_status("stopped", self.state.step)
                        return RunResult(self, stopped=True, answers=None)
                    state = self.state
                    state.grid = step(automaton, state.grid, state.rng, state.step)
                    state.step += 1
                    self._record(writers)
                    if state.step % self.experiment.checkpoint_every == 0:
                        self._checkpoint(writers)
                    if on_step is not None:
                        on_step(state)
                self._checkpoint(writers)
        except CellabError:
            self.folder.write_status("failed", self.state.step)
            raise
        answers = self.answers()
        self.folder.write_answers([answer.to_dict() for answer in answers])
        self.folder.write_status("finished", self.state.step)
        return RunResult(self, stopped=False, answers=answers)

    def _should_continue(self) -> bool:
        state, experiment = self.state, self.experiment
        if state.step < experiment.steps:
            return True
        if state.step >= experiment.max_steps:
            return False
        deterministic = experiment.automaton.deterministic
        return not all(q.resolved(state.history, deterministic) for q in experiment.questions)

    def _record(self, writers: dict[str, RecordWriter]) -> None:
        """Fingerprint the current grid and store what needs storing."""
        state = self.state
        fp = fingerprint(state.grid)
        state.history.record(fp)
        writers["fingerprints"].append(fp)
        if "generations" in writers:
            writers["generations"].append(state.grid.tobytes())
        if any(state.step in q.capture_steps() for q in self.experiment.questions):
            state.captured[state.step] = state.grid.copy()

    def _checkpoint(self, writers: dict[str, RecordWriter]) -> None:
        for writer in writers.values():
            writer.flush()
        self.folder.save_checkpoint(self.state, self.experiment.automaton.digest)
        self.folder.write_status("running", self.state.step)

    @contextmanager
    def _writers(self):
        writers = {"fingerprints": RecordWriter(self.folder.file(RunFolder.FINGERPRINTS))}
        if self.experiment.store_generations:
            writers["generations"] = RecordWriter(self.folder.file(RunFolder.GENERATIONS))
        try:
            yield writers
        finally:
            for writer in writers.values():
                writer.close()

    # ------------------------------------------------------------ reading results
    def grid_at(self, step_number: int) -> np.ndarray:
        """Any generation up to the current step: kept, stored on disk, or recomputed."""
        state, experiment = self.state, self.experiment
        if not 0 <= step_number <= state.step:
            raise ValueError(f"step {step_number} is outside 0..{state.step}")
        if step_number == state.step:
            return state.grid
        if step_number in state.captured:
            return state.captured[step_number]
        if experiment.store_generations:
            return self.folder.read_generation(step_number, state.grid.shape, state.grid.dtype)
        # Recompute from the start: same seed, same initial world, same steps.
        rng = make_generator(state.seed)
        grid = make_initial_grid(experiment, rng)
        for k in range(step_number):
            grid = step(experiment.automaton, grid, rng, k)
        return grid

    def answers(self) -> list[Answer]:
        context = AnswerContext(
            history=self.state.history, captured=self.state.captured, grid_at=self.grid_at,
            states=self.experiment.automaton.states, legend=self.experiment.legend,
            deterministic=self.experiment.automaton.deterministic, max_steps=self.experiment.max_steps,
        )
        return [question.answer(context) for question in self.experiment.questions]


class _StopFlag:
    requested = False


@contextmanager
def _stop_requests():
    """While active, Ctrl+C sets a flag instead of interrupting the program mid-step."""
    flag = _StopFlag()

    def handler(signum, frame):
        flag.requested = True

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:            # not in the main thread: leave Ctrl+C alone
        previous = None
    try:
        yield flag
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
