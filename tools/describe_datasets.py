"""Write docs/DATASETS.md: what every dataset in data/ contains, measured from the files themselves.

    python tools/describe_datasets.py
"""
from pathlib import Path

import numpy as np

from cellab.dataset import UNKNOWN, load_dataset

ROOT = Path(__file__).resolve().parent.parent
HEAD = """# The datasets

Made with `cellab dataset datasets/<name>.json` (see [FILE_FORMATS.md](FILE_FORMATS.md#dataset-file)); this page is
written by `python tools/describe_datasets.py` from the files in `data/`, so the numbers are measured, not typed.

**How they were made.** Every sample is one run of the ordinary CelLab engine from its own seed
(`dataset seed * 1,000,003 + sample number`), so each can be reproduced alone and the whole set is
identical every time it is generated. The labels are the facts `cellab run` reports for the five
questions; a test compares them with the answers of ordinary runs.

**What a sample holds.** Generation 0; the grid and the count of every state after each horizon N;
where the cycle starts and how long it is; from which step the grid is stable; and two
"does configuration C occur?" queries, one taken from the run (it occurs) and one with one or two
cells changed (usually it does not). `-1` always means "the run did not settle this".
`<name>.json` next to each file describes every array.

**Splits** (made when training, in `ml/data.py`): of each setting's samples 80% train, 10% validation,
10% test. For the elementary rules, 15% of the 88 rule families of every Wolfram class are kept out of
training completely and form the "unseen" split.

**Known limits.** Grids are small (16 cells, or 16 x 16) so that cycles can be found and labels are
exact. Automata with random rules give one random outcome per sample, so their targets are noisy by
nature and cycle answers do not exist for them. Starting grids are random with a few densities, which
is not how people draw grids by hand.

"""


def share(mask) -> str:
    return f"{100 * np.mean(mask):.0f}%"


def main() -> None:
    lines = [HEAD, "| Dataset | Automaton | Samples | Grid | Horizons | Rules | Stable | Cycle found | Median cycle | C occurs: yes / no / unknown | File |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    total = 0
    for path in sorted((ROOT / "data").glob("*.npz")):
        arrays, meta = load_dataset(path)
        spec, info = meta["spec"], meta["automaton"]
        found = arrays["cycle_length"] != UNKNOWN
        median = f"{int(np.median(arrays['cycle_length'][found])):,}" if found.any() else "–"
        occurs = arrays["occurs"]
        total += meta["samples"]
        lines.append(f"| `{meta['name']}` | {info['name']} | {meta['samples']:,} | {' x '.join(map(str, spec['size']))} | "
                     f"{', '.join(map(str, spec['horizons']))} | {'exact' if info['deterministic'] else 'random'} | "
                     f"{share(arrays['stable_from'] != UNKNOWN)} | {share(found)} | {median} | "
                     f"{share(occurs == 1)} / {share(occurs == 0)} / {share(occurs == UNKNOWN)} | {path.stat().st_size / 1e6:.1f} MB |")
    lines.append(f"\n{total:,} samples in all.\n")
    for path in sorted((ROOT / "data").glob("*.npz")):
        meta = load_dataset(path)[1]
        if meta["description"]:
            lines.append(f"- **`{meta['name']}`:** {meta['description']}")
    (ROOT / "docs" / "DATASETS.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote docs/DATASETS.md ({total:,} samples)")


if __name__ == "__main__":
    main()
