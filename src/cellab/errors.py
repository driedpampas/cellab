"""Error types.

Every problem a user can cause is reported as a CellabError with a message that
names the file and the field at fault, so the command line can print it as is.
"""

from pathlib import Path


class CellabError(Exception):
    """Base class for all errors CelLab reports to the user."""


class InputFileError(CellabError):
    """A problem in an automaton or experiment file.

    The message reads like `forest.json, rules[1].when: unknown name 'X'`.
    """

    def __init__(self, file: str | Path, location: str, message: str):
        self.file = str(file)
        self.location = location
        self.message = message
        where = f"{self.file}, {location}" if location else self.file
        super().__init__(f"{where}: {message}")


class RuleError(CellabError):
    """A rule failed while the automaton was running.

    Example: a rule produced a value that is not one of the declared states.
    """


class StorageError(CellabError):
    """A run folder is missing, damaged, or no longer matches its files."""
