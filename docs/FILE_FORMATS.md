# CelLab input files: reference

CelLab reads two kinds of JSON files:

- an **automaton file** (`automata/*.json`) says *what the automaton is*: its states, neighbourhood, edges and rules;
- an **experiment file** (`experiments/*.json`) says *what to do with it*: grid size, first generation, number of steps, seed, questions.

Check any file without running it:

```bash
cellab check automata/my_automaton.json experiments/my_experiment.json
```

Every mistake is reported with the file, the field and the reason, for example
`forest_fire.json, rules[1].when: 'self == FIER': unknown name 'FIER' (did you mean 'FIRE'?)`.

---

## Automaton file

```json
{
  "name": "Forest fire",
  "description": "Optional free text.",
  "dimensions": 2,
  "states": ["EMPTY", "TREE", "FIRE"],
  "neighbourhood": "von_neumann",
  "edges": {"type": "fixed", "value": "EMPTY"},
  "update": "synchronous",
  "parameters": {"p": 0.01},
  "rules": [
    {"when": "self == FIRE", "then": "EMPTY"},
    {"when": "self == TREE and count(FIRE) >= 1", "then": "FIRE"}
  ],
  "otherwise": "self",
  "colors": {"EMPTY": "#e3dfd4", "TREE": "#4a7326", "FIRE": "#d85a30"}
}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Display name. |
| `description` | no | Free text. |
| `dimensions` | yes | `1` (a row) or `2` (a grid). |
| `states` | yes | The values a cell can take (see below). |
| `neighbourhood` | yes | Which cells count as neighbours (see below). |
| `edges` | yes | What lies beyond the grid: `{"type": "wrap"}` (opposite sides touch) or `{"type": "fixed", "value": <state>}`. |
| `update` | no | How cells are updated. Only `"synchronous"` exists today: every cell is updated at once from the old grid. `"movement"` and `"agents"` are reserved for later. |
| `parameters` | no | Named numbers that rules can use. An experiment can override them. |
| `rules` | yes | Ordered list. For each cell, the **first rule that applies** decides its new value. |
| `otherwise` | no | The value of cells no rule applies to. Default `"self"` (unchanged). |
| `colors` | no | Colours for pictures (`"#rrggbb"`); ignored by the simulation. |

### States
| Form | Example | Notes |
|---|---|---|
| Named | `["OFF", "FIRING", "DYING"]` | Names are words (letters, digits, `_`). Stored as 0, 1, 2… in file order. |
| Integer | `{"type": "integer", "min": 0, "max": 3}` | `min` and `max` are optional. |
| Real | `{"type": "real", "min": 0, "decimals": 2}` | `decimals` rounds every new value, which lets decimal grids settle exactly. |

### Neighbourhoods
| Value | Cells |
|---|---|
| `"elementary"` | 1D: left and right. |
| `"von_neumann"` | 2D: the 4 cells above, left, right, below. |
| `"moore"` | 2D: all 8 surrounding cells. |
| `[[-1, 0], [0, 2], ...]` | Any list of offsets (row, column), or `[offset]` in 1D. |

### Rules
**When/then rule:** `{"when": "<condition>", "then": "<new value>"}`. Without `when`, the rule applies to every cell.

**Table rule:** `{"table": {"111": 0, "110": 0, ...}}` lists patterns of *the neighbours plus the cell itself, in reading order* (for `"elementary"`: left, self, right) and the value each pattern gives. Patterns not listed fall through to the next rule. Write each cell as one character (`"110"`), or separate cells with spaces for named states (`"ON OFF ON"`). Tables need named states or integer states with both `min` and `max`.

### The rule language
Rules are small expressions in Python syntax, evaluated for every cell at once.

| Name | Meaning |
|---|---|
| `self` | The cell's current value. |
| `left`, `right` | (1D) The neighbour on that side. |
| `north`, `south`, `east`, `west`, `north_east`, `north_west`, `south_east`, `south_west` | (2D) One neighbour. |
| `step` | The number of the generation being read (0, 1, 2, …). |
| State names | e.g. `FIRE` (named states only). |
| Parameter names | e.g. `p`. |
| `nb` | Inside `count/sum/mean/min/max` only: each neighbour in turn. |

| Function | Meaning |
|---|---|
| `count(x)` | How many neighbours make `x` true. `count(FIRE)` is short for `count(nb == FIRE)`. |
| `sum(x)`, `mean(x)`, `min(x)`, `max(x)` | Of `x` over the neighbours, e.g. `sum(nb)`. |
| `min(a, b, ...)`, `max(a, b, ...)` | With two or more arguments: cell by cell. |
| `random()` | A number in [0, 1) from the run's seeded generator. |
| `abs(x)`, `floor(x)`, `round(x)`, `round(x, digits)` | As usual. |

Operators: `+ - * / // % **`, comparisons (chains like `2 <= count(A) <= 3` work), `and`, `or`, `not`, `x if c else y`, and bit operators `& | ^ ~ << >>`. State names are written without quotes. Whole-number states must stay whole: use `//` rather than `/`.

### Examples
```json
"rules": [{"then": "self - 4 * (self >= 4) + count(nb >= 4)"}]
```
Sandpile: a cell with 4 or more grains loses 4, and gains one per toppling neighbour.

```json
"rules": [{"when": "count(nb == (self + 1) % 4) >= 2", "then": "(self + 1) % 4"}]
```
Cyclic automaton: move to the next value if at least two neighbours already hold it.

```json
"rules": [{"then": "(rule >> (4 * left + 2 * self + right)) & 1"}]
```
Any elementary rule, chosen by the parameter `rule`.

---

## Experiment file

```json
{
  "name": "Forest fire, grid from the brief",
  "automaton": "../automata/forest_fire.json",
  "size": [4, 5],
  "initial": {"type": "text", "rows": ["TTTT.", "TT.TT", ".TFTT", "TTTTT"],
              "legend": {"T": "TREE", "F": "FIRE", ".": "EMPTY"}},
  "steps": 8,
  "seed": 42,
  "max_steps": 1000,
  "store_generations": true,
  "checkpoint_every": 100,
  "parameters": {},
  "questions": [
    {"ask": "state_after", "n": 2},
    {"ask": "stable"},
    {"ask": "occurs", "rows": ["T....", ".....", ".....", "....."]},
    {"ask": "count_after", "n": 3, "value": "TREE"},
    {"ask": "cycle"}
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Display name; also the name of the run folder. |
| `automaton` | yes | Path to the automaton file, relative to this file. |
| `size` | if not given by `initial` | `[cells]` in 1D, `[rows, columns]` in 2D. |
| `initial` | yes | Generation 0 (see below). |
| `steps` | yes | How many generations to simulate (and store). |
| `seed` | no | Seed for random numbers. If missing, one is chosen and written into the run folder. |
| `max_steps` | no | How far a run may go to answer questions (default 10,000). |
| `store_generations` | no | Keep every generation on disk (default `true`). Needed for pictures. |
| `checkpoint_every` | no | Save a checkpoint every N steps (default 100). |
| `parameters` | no | New values for the automaton's parameters, e.g. `{"rule": 110}`. |
| `questions` | no | The questions to answer (see below). |

### The initial world
| Type | Example |
|---|---|
| `text` | `{"type": "text", "rows": ["..F..", ".F.F."], "legend": {".": "OFF", "F": "FIRING"}}`. Rows without spaces: one character per cell; rows with spaces: one word per cell (`"0 12 3"`). Tokens not in the legend are read as states directly. |
| `random` | `{"type": "random", "weights": {"TREE": 0.6, "EMPTY": 0.4}}`. Drawn from the seeded generator. Needs `size`. |
| `single` | `{"type": "single", "value": 1, "background": 0, "position": [2, 3]}`. One cell set on a uniform background (default position: the centre). Needs `size`. |

### Questions
| `ask` | Extra fields | Answer |
|---|---|---|
| `state_after` | `n` | The grid after n steps. Works for any n once a deterministic run has entered a cycle. |
| `count_after` | `n`, `value` | How many cells hold `value` after n steps. |
| `stable` | – | The step from which nothing changes, or "never" (with the cycle that prevents it). |
| `occurs` | `rows` (+ optional `legend`) | The first step showing this exact grid, or a certain "never". |
| `cycle` | – | Where the repetition starts and how long the cycle is. |

A question that `max_steps` was not enough to settle is answered "Not decided". For automata with random rules, answers about stability and cycles say that they are not guarantees.
