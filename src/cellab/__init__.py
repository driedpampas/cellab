"""CelLab: describe cellular automata in files, simulate them, answer questions about them.

The main entry points:

    load_automaton(path)      read and check an automaton file
    load_experiment(path)     read and check an experiment file (and its automaton)
    Run.start(experiment)     begin a run in a new run folder
    Run.open(folder)          reopen a run folder to continue it
    run.advance()             simulate until done (or stopped), then answer the questions
    step(automaton, grid, rng, step_number)   one generation, for direct use
"""

from .automaton import Automaton, load_automaton
from .engine import step
from .errors import CellabError, InputFileError, RuleError, StorageError
from .experiment import Experiment, load_experiment, make_initial_grid
from .runner import Run, RunResult, RunState

__version__ = "0.2.0"

__all__ = [
    "Automaton", "load_automaton", "step", "Experiment", "load_experiment", "make_initial_grid",
    "Run", "RunResult", "RunState", "CellabError", "InputFileError", "RuleError", "StorageError",
]
