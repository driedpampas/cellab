"""Run folders: everything needed to look at, reproduce, or continue a run.

    runs/<experiment name>/<date_time>/
        automaton.json      exact copy of the automaton file used
        experiment.json     copy of the experiment file, with the seed actually used
                            (its "automaton" field points to the copy next to it)
        checkpoint.npz      the run's state at the last checkpoint (see below)
        fingerprints.bin    16 bytes per step, appended as the run goes
        generations.bin     every generation, appended as the run goes (if stored)
        answers.json        the answers, written when the run finishes
        status.json         "running", "stopped" or "finished", and the step reached

Generations and fingerprints are append-only files of fixed-size records:
generation k starts at byte k * (bytes per grid). Appending is cheap, any step
can be read back directly, and resuming just cuts the files back to the last
checkpoint. Checkpoints are written to a temporary file and then renamed over
the old one, so a crash can never leave a half-written checkpoint behind.

checkpoint.npz holds: step, grid, initial grid, seed, the random generator's
state, the grids kept for questions, and the SHA-256 of automaton.json (a resume
refuses to continue if that file was edited since).
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .errors import StorageError
from .fingerprints import FINGERPRINT_SIZE

CHECKPOINT_FORMAT = 1


class RunFolder:
    AUTOMATON = "automaton.json"
    EXPERIMENT = "experiment.json"
    CHECKPOINT = "checkpoint.npz"
    FINGERPRINTS = "fingerprints.bin"
    GENERATIONS = "generations.bin"
    ANSWERS = "answers.json"
    STATUS = "status.json"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def file(self, name: str) -> Path:
        return self.path / name

    # ------------------------------------------------------------ creating
    @classmethod
    def create(cls, experiment, seed: int, runs_dir: str | Path = "runs", folder: str | Path | None = None) -> "RunFolder":
        """Make a new run folder and copy the input files into it."""
        if folder is None:
            base = Path(runs_dir) / _slug(experiment.name)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path, n = base / stamp, 2
            while path.exists():
                path, n = base / f"{stamp}_{n}", n + 1
        else:
            path = Path(folder)
            if path.exists() and any(path.iterdir()):
                raise StorageError(f"{path} already exists and is not empty")
        path.mkdir(parents=True, exist_ok=True)

        run = cls(path)
        shutil.copyfile(experiment.automaton.source, run.file(cls.AUTOMATON))
        raw = dict(experiment.raw, automaton=cls.AUTOMATON, seed=seed, steps=experiment.steps)
        _write_json(run.file(cls.EXPERIMENT), raw)
        run.write_status("running", 0)
        return run

    def update_experiment(self, experiment) -> None:
        """Rewrite experiment.json (used when a resumed run is given more steps)."""
        raw = dict(experiment.raw, automaton=self.AUTOMATON)
        _write_json(self.file(self.EXPERIMENT), raw)

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, state, automaton_digest: str) -> None:
        captured_steps = sorted(state.captured)
        if captured_steps:
            captured = np.stack([state.captured[k] for k in captured_steps])
        else:
            captured = np.empty((0, *state.grid.shape), dtype=state.grid.dtype)
        temporary = self.file("checkpoint.tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle,
                     format=np.array(CHECKPOINT_FORMAT),
                     step=np.array(state.step),
                     grid=state.grid,
                     initial=state.initial,
                     seed=np.array(str(state.seed)),
                     rng_state=np.array(state.rng_state()),
                     captured_steps=np.array(captured_steps, dtype=np.int64),
                     captured_grids=captured,
                     automaton_digest=np.array(automaton_digest))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.file(self.CHECKPOINT))

    def load_checkpoint(self) -> dict:
        path = self.file(self.CHECKPOINT)
        if not path.is_file():
            raise StorageError(f"{self.path} has no {self.CHECKPOINT}; is it a run folder?")
        with np.load(path, allow_pickle=False) as data:
            if int(data["format"]) != CHECKPOINT_FORMAT:
                raise StorageError(f"{path} was written by an incompatible version of CelLab")
            return {
                "step": int(data["step"]),
                "grid": data["grid"].copy(),
                "initial": data["initial"].copy(),
                "seed": int(str(data["seed"])),
                "rng_state": str(data["rng_state"]),
                "captured": {int(k): g.copy() for k, g in zip(data["captured_steps"], data["captured_grids"])},
                "automaton_digest": str(data["automaton_digest"]),
            }

    # ------------------------------------------------------------ append-only records
    def records(self, name: str, record_size: int) -> int:
        """How many complete records a .bin file holds."""
        path = self.file(name)
        return path.stat().st_size // record_size if path.exists() else 0

    def truncate(self, name: str, record_size: int, count: int) -> None:
        """Cut a .bin file back to its first `count` records (used when resuming)."""
        path = self.file(name)
        if path.exists():
            with open(path, "r+b") as handle:
                handle.truncate(count * record_size)

    def read_fingerprints(self, count: int) -> list[bytes]:
        data = self.file(self.FINGERPRINTS).read_bytes()[: count * FINGERPRINT_SIZE]
        if len(data) != count * FINGERPRINT_SIZE:
            raise StorageError(f"{self.FINGERPRINTS} in {self.path} is shorter than the checkpoint says")
        return [data[i:i + FINGERPRINT_SIZE] for i in range(0, len(data), FINGERPRINT_SIZE)]

    def read_generation(self, step: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        size = int(np.prod(shape)) * dtype.itemsize
        with open(self.file(self.GENERATIONS), "rb") as handle:
            handle.seek(step * size)
            data = handle.read(size)
        if len(data) != size:
            raise StorageError(f"generation {step} is not stored in {self.path}")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    def read_all_generations(self, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        path = self.file(self.GENERATIONS)
        if not path.is_file():
            raise StorageError(f"{self.path} has no stored generations (store_generations was false)")
        data = np.fromfile(path, dtype=dtype)
        per_grid = int(np.prod(shape))
        return data[: len(data) // per_grid * per_grid].reshape(-1, *shape)

    # ------------------------------------------------------------ answers and status
    def write_answers(self, answers: list[dict]) -> None:
        _write_json(self.file(self.ANSWERS), answers)

    def read_answers(self) -> list[dict]:
        path = self.file(self.ANSWERS)
        if not path.is_file():
            raise StorageError(f"{self.path} has no answers yet (the run is not finished)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_status(self, status: str, step: int) -> None:
        _write_json(self.file(self.STATUS), {"status": status, "step": step,
                                             "updated": datetime.now().isoformat(timespec="seconds")})

    def read_status(self) -> dict:
        return json.loads(self.file(self.STATUS).read_text(encoding="utf-8"))


class RecordWriter:
    """Appends fixed-size records (grids or fingerprints) to a .bin file."""

    def __init__(self, path: Path):
        self.handle = open(path, "ab")

    def append(self, data: bytes) -> None:
        self.handle.write(data)

    def flush(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.flush()
        self.handle.close()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "run"


def _write_json(path: Path, data) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
