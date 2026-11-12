# CelLab code guide

This guide explains what every file does, where the code for each part lives, and why it is built this way. The input file formats are in [FILE_FORMATS.md](FILE_FORMATS.md).

## 1. The idea in one paragraph

An automaton is **data**, not code. An automaton file lists the states, the neighbourhood, the edges and the rules. The rules are small expressions such as `self == TREE and count(FIRE) >= 1`. One general engine reads any such file and computes the next generation for every cell at once with NumPy. Around the engine, a runner repeats the step, takes a **fingerprint** (hash) of every generation, saves **checkpoints** so a run can be stopped and resumed, and answers the **five questions** from the fingerprints. Randomness comes from **one seeded generator** per run, so the same seed always gives the same run.

## 2. The layers

```
 automata/*.json    experiments/*.json                         INPUT FILES
        │                   │
        ▼                   ▼
 ┌──────────────────────────────────────────────┐
 │ automaton.py   experiment.py                 │  1. LOADING: JSON → checked objects
 │ loading.py  states.py  grids.py              │     (all errors found here, before running)
 │ neighbourhood.py (resolve)                   │
 ├──────────────────────────────────────────────┤
 │ expressions.py   rules.py                    │  2. RULES: compiled rule objects
 ├──────────────────────────────────────────────┤
 │ engine.py   neighbourhood.py (views)         │  3. ONE STEP: old grid → new grid
 │ randomness.py                                │     (pure, apart from seeded random draws)
 ├──────────────────────────────────────────────┤
 │ runner.py   fingerprints.py   questions.py   │  4. A RUN: loop, fingerprints, answers
 │ storage.py                                   │     checkpoints, resume
 ├──────────────────────────────────────────────┤
 │ cli.py   render.py                           │  5. OUTSIDE: command line, pictures
 └──────────────────────────────────────────────┘
        │
        ▼
 runs/<experiment>/<date_time>/   automaton.json, experiment.json, checkpoint.npz,
                                  fingerprints.bin, generations.bin, answers.json, status.json
```

Each layer only uses the layers above it. The engine never reads files, the loaders never touch NumPy grids of a run, and the command line contains no logic of its own.

## 3. File by file (`src/cellab/`)

| File | Job | Main names |
|---|---|---|
| `errors.py` | The error types. `InputFileError` carries the file and the field, so every message reads `file, field: problem`. | `CellabError`, `InputFileError`, `RuleError`, `StorageError` |
| `loading.py` | Reads JSON files and turns pydantic's error lists into one readable message. | `read_json`, `validate`, `location` |
| `states.py` | What a cell may hold: named states, integers or reals. Converts file values to numbers, formats them back, and checks every new grid. | `StateSpace`, `state_space_from_file` |
| `grids.py` | Grids written as text rows (`"TT.F"` or `"0 12 3"`), with a legend. | `parse_grid`, `format_grid`, `split_row` |
| `neighbourhood.py` | Neighbourhood offsets, direction names, and `GridViews`: the padded grid that hands out shifted copies. | `resolve_neighbourhood`, `GridViews` |
| `expressions.py` | The rule language: checks expressions when files are loaded, evaluates them with NumPy during steps. | `compile_expression`, `Expression`, `EvalContext` |
| `rules.py` | The two rule kinds. Both return "applies here?" and "new value" for every cell. | `ExpressionRule`, `TableRule`, `build_table_rule` |
| `automaton.py` | Loads and checks an automaton file; compiles every rule. | `load_automaton`, `Automaton` |
| `experiment.py` | Loads and checks an experiment file; builds generation 0 and the question objects. | `load_experiment`, `Experiment`, `make_initial_grid` |
| `engine.py` | **One step**: old grid → new grid, for the synchronous mode. | `step` |
| `agents.py` | One step of the agents mode: decide for all cells at once, then act cell by cell. Also checks the `do` part of rules. | `AgentRule`, `Action`, `Choice`, `step` |
| `randomness.py` | Seeds, the generator, saving and restoring its state. | `make_generator`, `generator_state`, `restore_generator` |
| `fingerprints.py` | 16-byte hashes of grids, and the `History` that spots repeats and standstills. | `fingerprint`, `History` |
| `questions.py` | The five questions: what each needs, when it's resolved, how it's answered. | `StateAfter`, `CountAfter`, `Stable`, `Occurs`, `Cycle` |
| `storage.py` | Run folders: copies of the inputs, atomic checkpoints, append-only record files. | `RunFolder`, `RecordWriter` |
| `runner.py` | The loop: start, step, fingerprint, store, checkpoint, stop on Ctrl+C, resume, answer. | `Run`, `RunState`, `RunResult` |
| `render.py` | PNG, GIF and 1D space-time pictures of a run. | `render_run` |
| `dataset.py` | Machine-learning datasets: runs every sample with the engine and collects the exact answers; also writes them as questions and answers in words. | `build_dataset`, `run_sample`, `load_dataset`, `question_answer_pairs` |
| `cli.py` | The `cellab` command: `check`, `run`, `resume`, `answers`, `show`, `render`, `dataset`, `qa`. | `main` |

The neural networks live outside the package, in `ml/` (they need PyTorch; the simulator doesn't): `eca.py` (rule families and classes), `data.py` (loading and splits), `models.py` (the two networks), `train.py`, `evaluate.py`. See [deliverables/D3-dataset/README.md](deliverables/D3-dataset/README.md).

## 4. How each part works

### 4.1 Loading (automaton.py, experiment.py)
Loading happens in three layers, each with its own kind of message:

1. **JSON:** `read_json` reports the line and column of a syntax error.
2. **Structure:** a pydantic model (`_AutomatonFile`, `_ExperimentFile`) checks required fields, unknown fields (typos like `neighborhood`), and basic types.
3. **Meaning:** `build_automaton` / `load_experiment` check what pydantic can't:
   - states are valid and not reserved words
   - the edge value is a state
   - parameters don't clash with state names
   - every rule compiles, so an unknown name stops loading
   - tables have the right pattern length
   - the start grid and question grids fit the size, and so on

The result (`Automaton`, `Experiment`) contains no raw text any more. State names have become numbers, the neighbourhood is a tuple of offsets, and every rule is a compiled object.

### 4.2 The rule language (expressions.py)
- **Parsing:** Python's own parser reads the text (`ast.parse(..., mode="eval")`). `_Checker` walks the tree and accepts only a whitelist: numbers, names, arithmetic, comparisons, `and/or/not`, `if/else`, and the known functions. Anything else (attributes, imports, lambdas, lists, strings) is rejected with a message. That makes the language **safe**: a file can never run arbitrary Python.
- **Evaluation:** `_evaluate` walks the checked tree. Each name stands for a whole grid: `self` is the grid, `north` is the grid shifted by one row, `FIRE` is the number 2. So every operation works on all cells at once.
- **Aggregates:** `count/sum/mean/min/max(x)` evaluate `x` once per neighbour, with `nb` bound to that neighbour's shifted grid, stack the results, and reduce over the stack.
- **Random draws stay fixed:** `and`/`or` evaluate every part (no short-circuit), so the number and order of `random()` draws per step never depend on the data.
- **Runtime errors** (division by zero, …) become `RuleError`s that name the rule: `rules[0].then ('1 // (self - self)'): divide by zero`.

### 4.3 One step (engine.py)
```
values  = grid in a wide number type (no overflow surprises)
views   = GridViews(values, radius, edges)        # padded once
work    = value of `otherwise` for every cell     # usually "unchanged"
decided = all False
for each rule in file order:
    applies, new = rule.evaluate(ctx)             # two full-size arrays
    chosen = applies and not decided              # first matching rule wins
    work[chosen] = new[chosen]                    # (checked: whole numbers etc.)
    decided |= chosen
return states.finish_step(work)                   # rounding, limits, final type
```
- Every rule is evaluated for every cell, then masked. That's fast with NumPy, and keeps random draws in a fixed order.
- Everything reads the **old** grid, so cell order never matters: this is the synchronous update the brief describes.

**Padding** (`neighbourhood.GridViews`): the grid is padded by `radius` cells, copied from the other side (`wrap`) or set to the edge value (`fixed`). A neighbour at offset `(dr, dc)` is the slice `padded[r+dr : r+dr+H, r+dc : r+dc+W]`, a view with no copying and no loop over cells.

**Table rules** (`rules.TableRule`): each cell's pattern is turned into a number (reading the cells as digits in base *number of states*) and looked up in two arrays, `listed` and `results`, built when the file was loaded.

### 4.3b One step with agents (agents.py)
Five of the brief's automata (Langton's ant, falling sand, Schelling, predator–prey, ecosystem) need one thing the synchronous step can't do: a decision that changes **two** cells, with a rule for who gets a place when two want it.

```
old = the state as numbers: old[0] is the grid, old[1:] are the layers
1. decide (vectorised, from the old grid; the same EvalContext and GridViews as engine.py)
   rule_of[cell] = the first rule whose `when` holds there
   acting = those cells, in the automaton's order (reading, reverse, or shuffled with the run's generator)
2. act (a loop over the acting cells, on a live copy)
   skip the cell if another cell's action has written to it in this step
   for each action of its rule, in order:
       if the action has `to`: candidates = neighbours / listed directions / anywhere
                               keep those satisfying `where`; none left -> try the next action
                               take the first, or a seeded random one
       compute every value of `here` and `there`, then write them; mark the other cell as written
       stop at the first action that happened
```
- **Layers** are how a cell holds more than its state. A state of a layered automaton is one array `(1 + number of layers, rows, columns)`; `Automaton.cells(state)` and `Automaton.layer(state, name)` take it apart. Everything outside the step (runner, storage, checkpoints) just sees "the state array", so it needed no special cases.
- **`_CellContext`** gives expressions the live values of one cell through the same interface as `EvalContext`, so the rule language is evaluated by the same code in both modes.
- **Fingerprints:** a layered state gets two per step: one of the whole state (for repeats and standstills, where an ant's heading matters) and one of the grid alone (for "does this configuration occur?"). Automata without layers are untouched: a test pins their fingerprints to the values shown in Deliverable 2.
- **Speed:** deciding is as fast as a synchronous step; acting costs about 60 µs per acting cell. Langton's ant runs at about 3,000 steps per second, a 30×30 predator–prey world at about 20.

### 4.4 Fingerprints and the five questions (fingerprints.py, questions.py)
- `fingerprint(grid)` is a 16-byte BLAKE2b hash of the grid's shape, type and bytes. Two different grids collide with probability about 2⁻¹²⁸.
- `History.record` keeps:
  - `first_seen[fp] = first step`
  - `repeat = (earlier, later)` the first time a fingerprint comes back
  - `standstill`: the first step equal to the step before
- **Key fact:** for deterministic rules, a repeat means the run is in a cycle forever, so it decides every question at once:

| Question | How it is answered |
|---|---|
| `state_after(n)` | The kept grid of step n. Beyond the run, inside a cycle: the grid of step `start + (n − start) mod length`. |
| `count_after(n, X)` | The same grid, then count X. |
| `stable` | `standstill − 1`; or "never", if the cycle has length > 1. |
| `cycle` | `repeat` gives the start and the length. |
| `occurs(C)` | Look up C's fingerprint in `first_seen`. Once the cycle has closed without it: a certain "never". |

- Each question also says which grids it needs kept (`capture_steps`) and whether the run has gathered enough (`resolved`). The runner keeps going until the experiment's `steps` are done and every question is resolved, or `max_steps` is reached. Unresolved questions are answered "Not decided".
- **Random rules** have no such guarantee: the answers say so instead of claiming a cycle.

### 4.5 Fixed randomness (randomness.py)
- Each run has **one** `numpy.random.Generator(PCG64(seed))`. Nothing else in the code makes random numbers; a test checks the source code for it.
- **Order of draws:** first the random initial world (if any), then every step one full-grid array per `random()` in the rules, in rule order.
- **Missing seed:** if the experiment has none, `new_seed()` picks one and the run folder records it, so any run can be repeated.
- **Saved state:** the generator's internal state is saved in every checkpoint, so a resumed run continues the same sequence.

### 4.6 Saving and resuming (storage.py, runner.py)
**A run folder** holds:
- copies of both input files (with the real seed written in)
- `checkpoint.npz`: step, grid, initial grid, seed, generator state, the kept grids, and the SHA-256 of `automaton.json`
- two **append-only** files of fixed-size records: `fingerprints.bin` (16 bytes per step) and `generations.bin` (one grid per step). Generation k starts at byte `k × grid size`, so any step can be read directly.

**Checkpoints are atomic:** written to `checkpoint.tmp`, then renamed over `checkpoint.npz`. A crash leaves either the old or the new checkpoint, never half of one.

**Ctrl+C** only sets a flag (`runner._stop_requests`). The loop notices it between two steps, saves a checkpoint and stops cleanly. `--stop-at N` does the same at a chosen step.

**Resume** (`Run.open`):
1. Load the folder's own copies of the files.
2. Refuse if `automaton.json` changed (its digest).
3. Rebuild the `History` from `fingerprints.bin`.
4. Cut both `.bin` files back to the checkpoint, dropping steps computed after a crash.
5. Restore the generator, and continue with the same loop.

The tests check that the result is byte-for-byte the same as an uninterrupted run. They cover stops at several steps, Ctrl+C, and a simulated crash between checkpoints.

**Trade-off:** each checkpoint forces the files to disk (`fsync`), which is slow on Windows. That's why the default is every 100 steps: at most 100 steps are recomputed after a crash.

### 4.7 The command line (cli.py)
Each command parses arguments and calls the library; there's no logic in `cli.py` itself. Library errors (`CellabError`) are printed as one line starting with `error:`, never as a traceback.

## 5. Where to change things later

| Future need | Where it goes | What does *not* change |
|---|---|---|
| A new automaton | A new JSON file | Any code |
| A new function in rules (e.g. `sqrt`) | `expressions.py`: the checker (`check_call`) and `_call` | Everything else |
| A new neighbourhood name | `NAMED_NEIGHBOURHOODS` in `neighbourhood.py` | Everything else |
| A new kind of action for agents | `agents.py`: the `_Action` model, `build_agent_rule` and `_World.perform` | The synchronous mode, storage, questions |
| A third update mode | `engine.step` chooses the mode named in the file's `update` field | Runner, fingerprints, questions, storage, CLI: they only see state arrays |
| A new dataset | A new file in `datasets/`, then `cellab dataset` | Any code |
| A new network or training method | `ml/models.py`, `ml/train.py` | The package `cellab` |
| Deliverable 4 parser | Produce an automaton JSON file; `load_automaton` validates it | Everything else |
| A graphical interface | Call `load_experiment`, `Run.start`, `run.advance(on_step=...)`, `render` | The library |
| Changing the checkpoint contents | Raise `CHECKPOINT_FORMAT` in `storage.py` | Old runs are refused with a clear message |

## 6. Tests (`tests/`)

Run them with `python -m pytest`; all 293 run in about 75 seconds (35 of them need PyTorch and are skipped without it).

| File | What it proves |
|---|---|
| `test_engine_reference.py` | The engine agrees with **independent, naive implementations** (`reference.py`: plain loops, no shared code) on random grids, for every example automaton and 7 elementary rules. |
| `test_known_results.py` | Results known in advance: the brief's forest fire, sandpile and Rule 58 example; the Rule 30 cycle table and dataset rows from the Deliverable 1 slides; Rule 90 dying on 16 cells; traffic keeping its cars; the glider's 48-step cycle. |
| `test_rule_language.py` | Every operator, function, direction and edge type on small grids; first-rule-wins; runtime errors (out-of-range values, fractions in whole-number states, division by zero). |
| `test_validation.py` | Over 50 broken files are each rejected with the right field and message. |
| `test_reproducibility.py` | Same seed gives the same run; different seeds differ; the seed is recorded; no other randomness exists in the code. Stopping (at several steps, by Ctrl+C, or a crash) and resuming equals an uninterrupted run, byte for byte. An edited automaton is refused. |
| `test_questions.py` | Jump-ahead through a cycle (checked against 12,345 directly simulated steps), "never", "not decided", random rules, saved answers. |
| `test_generality.py` | **No state name from any example appears in the code**; every example file loads and runs. |
| `test_agents.py` | The agents mode on tiny hand-made automata (turns, order, fallbacks, layers, `anywhere`); Langton's ant and falling sand against independent implementations; conservation and rule checks for Schelling, predator–prey and the ecosystem; over 20 broken agent files; resuming and fingerprints with layers. |
| `test_dataset.py` | Dataset labels equal the answers of ordinary runs; the same file always gives the same dataset, with any number of worker processes; unknown answers are marked; questions and answers in words. |
| `test_ml.py` | (Needs PyTorch.) The 88 rule families; whole families are kept out of training; both networks in 1D and 2D; the one-step network learns and simulates unseen rules; training resumes and is evaluated. |
| `test_cli_and_storage.py` | Every command; run folder contents; generations on disk; replay without stored generations; pictures. |

## 7. Glossary
- **Generation / step:** generation 0 is the initial world; step k produces generation k.
- **Synchronous update:** every cell's new value is computed from the old grid; the order of cells doesn't matter.
- **Fingerprint:** a 16-byte hash identifying a grid.
- **Cycle (start s, length L):** generation s + L equals generation s, and then everything repeats.
- **Checkpoint:** a saved copy of everything that changes during a run.
- **Run folder:** a self-contained record of one run, enough to replay, continue or draw it.
