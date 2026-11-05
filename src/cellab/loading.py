"""Shared helpers for reading the JSON input files.

`read_json` gives clear errors for missing files and broken JSON (with the line and
column). `validate` runs a pydantic model over the data and turns its error list
into one readable InputFileError, e.g.

    forest.json, rules[0].then: this field is required
"""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .errors import InputFileError


def read_json(path: str | Path) -> dict:
    """Read a JSON file that must contain one object { ... }."""
    path = Path(path)
    if not path.is_file():
        raise InputFileError(path, "", "file not found")
    try:
        text = path.read_text(encoding="utf-8-sig")   # -sig: also accept the BOM Windows editors add
    except UnicodeDecodeError:
        raise InputFileError(path, "", "the file is not UTF-8 text") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise InputFileError(path, f"line {error.lineno}, column {error.colno}",
                             f"invalid JSON ({error.msg})") from None
    if not isinstance(data, dict):
        raise InputFileError(path, "", "the file must contain a JSON object { ... }")
    return data


def location(parts, prefix: str = "") -> str:
    """('rules', 1, 'when') -> 'rules[1].when'"""
    text = prefix
    for part in parts:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else str(part)
    return text


def validate(model: type[BaseModel], data, file: str | Path, prefix: str = ""):
    """Check `data` against a pydantic model and return the model instance."""
    try:
        return model.model_validate(data)
    except ValidationError as error:
        lines = []
        # Unknown fields first: a misspelled field usually also causes a "required" error.
        problems = sorted(error.errors(), key=lambda problem: problem["type"] != "extra_forbidden")
        for problem in problems:
            where = location(problem["loc"], prefix) or "(top level)"
            if problem["type"] == "missing":
                message = "this field is required"
            elif problem["type"] == "extra_forbidden":
                allowed = ", ".join(model.model_fields) if not problem["loc"][:-1] else ""
                message = "unknown field (check the spelling)" + (f"; allowed fields: {allowed}" if allowed else "")
            else:
                message = problem["msg"][0].lower() + problem["msg"][1:]
            lines.append((where, message))
        first_where, first_message = lines[0]
        extra = "".join(f"\n  also {where}: {message}" for where, message in lines[1:])
        raise InputFileError(file, first_where, first_message + extra) from None
