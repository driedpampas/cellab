"""Pictures of runs. Only for people: the engine never looks at colours.

    2D automata   first.png, last.png and run.gif (an animation of the generations)
    1D automata   spacetime.png: one row per generation, time going down

Colours come from the automaton's optional "colors" field. Values without a
colour get one from a default palette (named states) or a light-to-dark shade
(numbers, scaled between the smallest and largest value in the run).
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .automaton import Automaton
from .experiment import load_experiment
from .storage import RunFolder

DEFAULT_PALETTE = [(245, 243, 238), (22, 32, 43), (216, 90, 48), (74, 115, 38), (43, 92, 153),
                   (239, 159, 39), (153, 53, 86), (29, 158, 117), (136, 135, 128), (127, 119, 221)]
LIGHT, DARK = np.array([245, 243, 238], float), np.array([22, 32, 43], float)
MAX_GIF_FRAMES = 300


def painter(automaton: Automaton, values: np.ndarray):
    """A function that turns a grid (or a stack of grids) into RGB colours."""
    states = automaton.states
    if states.kind == "named":
        table = np.array([automaton.colors.get(i, DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
                          for i in range(len(states.names))], dtype=np.uint8)
        return lambda grid: table[grid]

    low = float(values.min()) if states.minimum is None else float(states.minimum)
    high = float(values.max()) if states.maximum is None else float(states.maximum)
    span = (high - low) or 1.0

    def paint(grid):
        share = np.clip((grid.astype(float) - low) / span, 0, 1)[..., np.newaxis]
        rgb = (LIGHT + share * (DARK - LIGHT)).astype(np.uint8)
        for value, colour in automaton.colors.items():
            rgb[grid == value] = colour
        return rgb

    return paint


def to_image(rgb: np.ndarray, cell: int) -> Image.Image:
    if rgb.ndim == 2:                       # a single 1D row
        rgb = rgb[np.newaxis]
    height, width = rgb.shape[:2]
    return Image.fromarray(rgb, "RGB").resize((width * cell, height * cell), Image.NEAREST)


def cell_size(height: int, width: int, target: int = 480) -> int:
    return max(1, min(24, target // max(height, width)))


def render_run(path: str | Path, out_dir: str | Path | None = None, frame_ms: int = 120) -> list[Path]:
    """Make pictures of a run folder's stored generations. Returns the files written."""
    folder = RunFolder(path)
    experiment = load_experiment(folder.file(RunFolder.EXPERIMENT))
    automaton = experiment.automaton
    generations = folder.read_all_generations(experiment.shape, automaton.states.dtype)
    out = Path(out_dir) if out_dir else folder.path
    out.mkdir(parents=True, exist_ok=True)
    paint = painter(automaton, generations)
    written = []

    if automaton.dimensions == 1:
        rgb = paint(generations)                                   # (time, cells, 3)
        image = to_image(rgb, cell_size(*rgb.shape[:2], target=720))
        image.save(out / "spacetime.png")
        return [out / "spacetime.png"]

    cell = cell_size(*experiment.shape)
    for name, grid in (("first.png", generations[0]), ("last.png", generations[-1])):
        to_image(paint(grid), cell).save(out / name)
        written.append(out / name)

    chosen = generations
    if len(generations) > MAX_GIF_FRAMES:
        chosen = generations[np.linspace(0, len(generations) - 1, MAX_GIF_FRAMES).astype(int)]
    frames = [to_image(paint(grid), cell) for grid in chosen]
    frames[0].save(out / "run.gif", save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
    written.append(out / "run.gif")
    return written
