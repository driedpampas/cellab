"""Allows `python -m cellab ...` as well as `cellab ...`."""

from .cli import main

raise SystemExit(main())
