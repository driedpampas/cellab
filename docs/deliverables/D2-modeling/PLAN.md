# Deliverable 2 plan: modelling automata in CelLab

Lab 6 deliverable, due next week. Written 29 October 2026 from the brief, the Lab 5 notes and our Deliverable 1 plan.

---

## 1. What Deliverable 2 must contain

**From the schedule:** "Modeling an automaton within your system: world state and rules, experiment parameters, state storage and recovery."

**From the Lab 5 notes** (`docs/requirements/lab5-notes-deliverable2.png`, translated):

| Note (Romanian) | Meaning | How we meet it |
|---|---|---|
| reguli citite din fișiere | Rules are read from files | Each automaton is a spec file (section 4) |
| soluții păstrat starea automatului celular | A way to save the automaton's state | Run folders with checkpoints (section 9) |
| reîncărcați – reîncepe de unde a rămas | Reload and continue from where it stopped | `cellab resume` gives the same result as an uninterrupted run (section 9) |
| menționa parametrii – cele 5 întrebări | State the parameters, including those of the 5 questions | Experiment file with steps, seed, grid, initial world and the 5 questions (section 5) |
| nimic specific unui automat în cod – generalist + depinde de input din fișier | No automaton-specific code; general engine driven by the input file | Expression-based rules plus a test that fails if automaton names appear in the code (section 13) |
| fișiere de input | Input files | `automata/` and `experiments/` folders |
| 2–3 automate din cele 15 (aș face mai multe) | 2–3 of the 15 automata (the professor would do more) | 3 required, 10+ planned (section 11) |
| hash-uri peste stări → punctaj maxim | Hashes over states, for the maximum score | Fingerprint of every state, used for the questions (section 7) |
| fixați randomness – rulați de 2 ori același seed, același rezultat → punctaj maxim | Fix the randomness: same seed twice gives the same result, for the maximum score | One seeded generator saved in every checkpoint, plus tests (section 8) |

### Definition of done (checklist for the lab)
- [ ] Automata are defined only by input files; the engine contains no automaton-specific code (checked by a test)
- [ ] At least 3 automata from the brief work; target 10 or more
- [ ] Experiment parameters (grid, initial world, steps, seed, storage options, the 5 questions with their parameters) come from a file
- [ ] The state can be saved, reloaded, and the run continued; resuming gives exactly the same result as never stopping
- [ ] Every state is fingerprinted with a hash; the 5 questions are answered using the fingerprints
- [ ] Same seed twice gives identical runs, shown live and by tests
- [ ] Invalid input files are rejected with clear messages
- [ ] Tests pass; README explains installation and usage
- [ ] AI conversations saved (both of us)
- [ ] Short presentation + live demo ready

---

## 2. Long-term goals to keep in mind

From `docs/requirements/lab5-notes-final-presentation.png` (the final presentation):

- a PowerPoint presentation that is **fun and creative**, even **a bit funny**
- an **interesting interface**
- **not only exactly what is asked**: a good project that **stands out**
- a **good research part**
- all criteria met; aiming for a **10**, by being **very creative**

What this means for Deliverable 2:
- Build the core cleanly; everything later (interface, dataset, parser) stands on it.
- Keep visual output from day one (images and GIFs of runs). It is cheap, and it becomes material for the final presentation.
- Start the research part early: the engine can already measure cycle statistics for all 256 elementary rules (stretch goal, section 16), which previews the Deliverable 3 experiment.
- Make the Deliverable 2 demo memorable: live, short, with a couple of moments people remember (section 15).

---

## 3. Design overview

Two kinds of input files, kept separate:

- **Automaton spec** (`automata/*.json`): *what the automaton is*. States, neighbourhood, edges, rules, rule parameters.
- **Experiment file** (`experiments/*.json`): *what we do with it*. Grid size, initial world, number of steps, seed, storage options, the questions to answer.

One automaton can be used in many experiments, and an experiment can be rerun exactly.

```
automaton spec ─┐
                ├─► validator ─► engine ─► fingerprints ─► answers to the 5 questions
experiment file ┘                  │
                                   └─► run folder: checkpoints, stored generations, answers, images
                                          ▲
                              cellab resume ┘  (continues from the last checkpoint)
```

---

## 4. The automaton spec format

JSON, because the Deliverable 4 parser will produce JSON and it needs no extra library.

### Structure
| Field | Meaning | Values |
|---|---|---|
| `name` | Display name | text |
| `dimensions` | 1 or 2 | `1`, `2` |
| `states` | Possible cell values | a list of names (`["EMPTY","TREE","FIRE"]`), or `{"type": "integer", "min": 0}`, or `{"type": "real"}` |
| `neighbourhood` | Which cells a cell sees | `"elementary"` (left, self, right), `"von_neumann"` (4), `"moore"` (8), or a list of offsets |
| `edges` | What lies beyond the grid | `{"type": "wrap"}` or `{"type": "fixed", "value": ...}` |
| `parameters` | Named numbers used in rules | e.g. `{"p": 0.01}` |
| `rules` | Ordered list; the first matching rule wins | `{"when": <condition>, "then": <new value>}` or a pattern `table` |
| `otherwise` | Value when no rule matches | default `"self"` (unchanged) |

### The rule language
Rules are small expressions, evaluated for the whole grid at once with NumPy:

- **Cell values:** `self`; in 1D also `left` and `right`.
- **Neighbour aggregates:** `count(<condition on nb>)`, `sum(nb)`, `mean(nb)`, `min(nb)`, `max(nb)`. Inside them, `nb` stands for each neighbour in turn. `count(FIRE)` is short for `count(nb == FIRE)`.
- **Names:** declared state names and `parameters`.
- **Operators:** `+ - * / // % **`, comparisons, `and or not`, `x if c else y`.
- **Functions:** `random()` (seeded, section 8), `abs`, `round`, `floor`.

**How it's implemented:** Python's own `ast` module parses each expression. Only a whitelist of node types is allowed, so the input can't run arbitrary code. The expression is then compiled into NumPy operations, so we don't write a parser from scratch.

**Rule semantics:** start from `otherwise`, then apply the rules from last to first with `np.where`, so earlier rules take precedence.

### Examples
**Forest fire** (deterministic, from the brief):
```json
{
  "name": "Forest fire",
  "dimensions": 2,
  "states": ["EMPTY", "TREE", "FIRE"],
  "neighbourhood": "von_neumann",
  "edges": {"type": "fixed", "value": "EMPTY"},
  "rules": [
    {"when": "self == FIRE", "then": "EMPTY"},
    {"when": "self == TREE and count(FIRE) >= 1", "then": "FIRE"}
  ],
  "otherwise": "self"
}
```

**Forest fire with regrowth and lightning** (random; the same file plus parameters):
```json
"parameters": {"p": 0.01, "f": 0.0001},
"rules": [
  {"when": "self == FIRE", "then": "EMPTY"},
  {"when": "self == TREE and (count(FIRE) >= 1 or random() < f)", "then": "FIRE"},
  {"when": "self == EMPTY and random() < p", "then": "TREE"}
]
```

**Rule 30** (the brief's truth table, written as a pattern table of left, self, right):
```json
{
  "name": "Rule 30",
  "dimensions": 1,
  "states": [0, 1],
  "neighbourhood": "elementary",
  "edges": {"type": "fixed", "value": 0},
  "rules": [{"table": {"111": 0, "110": 0, "101": 0, "100": 1,
                       "011": 1, "010": 1, "001": 1, "000": 0}}]
}
```

**Brian's Brain:**
```json
"states": ["OFF", "FIRING", "DYING"],
"neighbourhood": "moore",
"rules": [
  {"when": "self == FIRING", "then": "DYING"},
  {"when": "self == DYING", "then": "OFF"},
  {"when": "self == OFF and count(FIRING) == 2", "then": "FIRING"}
]
```

**Sandpile** (numeric; all unstable cells topple at once, grains falling off the edge are lost):
```json
"states": {"type": "integer", "min": 0},
"neighbourhood": "von_neumann",
"edges": {"type": "fixed", "value": 0},
"rules": [{"then": "self - 4 * (self >= 4) + count(nb >= 4)"}]
```

**Cyclic, 4 states:**
```json
"states": {"type": "integer", "min": 0, "max": 3},
"neighbourhood": "moore",
"rules": [{"when": "count(nb == (self + 1) % 4) >= 2", "then": "(self + 1) % 4"}]
```

The same language covers voter, crystal growth, epidemic (infection timers as states I1, I2, I3), diffusion, traffic (Rule 184) and Game of Life, all without new code.

---

## 5. The experiment file (parameters and the 5 questions)

```json
{
  "name": "Forest fire, grid from the brief",
  "automaton": "../automata/forest_fire.json",
  "grid": {"width": 5, "height": 4},
  "initial": {
    "type": "text",
    "rows": ["TTTT.", "TT.TT", ".TFTT", "TTTTT"],
    "legend": {"T": "TREE", "F": "FIRE", ".": "EMPTY"}
  },
  "steps": 20,
  "max_steps": 100000,
  "seed": 42,
  "store_generations": true,
  "checkpoint_every": 10,
  "questions": [
    {"ask": "state_after", "n": 2},
    {"ask": "stable_when"},
    {"ask": "occurs", "configuration": {"rows": ["T....", ".....", ".....", "....."]}},
    {"ask": "count_after", "state": "TREE", "n": 3},
    {"ask": "cycle"}
  ]
}
```

**Initial world types:**
- `text`: rows with a legend
- `random`: with `weights` per state, e.g. `{"TREE": 0.7, "EMPTY": 0.3}`
- `single`: one cell of a given state in the centre
- `file`: a saved grid or generation

`max_steps` is the budget for questions that may need a long run. The seed is always recorded: if the file has none, we generate one and write it into the run folder.

---

## 6. The engine

- **Grid storage:**
  - named states are stored as `uint8` indices
  - integer states as `int32`
  - real states as `float64`, with an optional rounding setting (the brief's diffusion example rounds to whole numbers)
- **Neighbourhoods** are lists of offsets. Edges are handled by padding the grid (`np.pad` with `wrap` or a constant). Each neighbour is then a shifted view of the padded array, so the code is the same for 1D and 2D and for any neighbourhood.
- **One step:** evaluate every rule's condition and value on the old grid, then build the new grid. All cells update at the same time, as the brief describes.
- **Update modes:** Deliverable 2 implements the synchronous mode, which covers 10–11 of the 15 examples. Movement (traffic with speeds, falling sand) and agents (Langton's ant, predator–prey, Schelling, ecosystem) come later. Their file format is designed now, so the presentation can show it extends (section 16).
- **Performance:** fully vectorised, so a 200 × 200 grid for 1,000 steps takes seconds.

---

## 7. Fingerprints (hashes) and the five questions

- **Fingerprint of a state:** `blake2b` (16 bytes) of the grid bytes plus its shape and data type. That's fast, and the chance of two different states colliding is negligible. On a match we still compare the stored grids when they are available.
- **Fingerprint table:** maps each fingerprint to the step where it first appeared, kept in memory and saved in every checkpoint. At 16 bytes per step, a million steps take 16 MB.
- **The five questions:**
  1. **State after N:** simulate to N. If a cycle is already known and N is huge, jump ahead: state(N) = state(μ + (N − μ) mod λ).
  2. **When is it stable:** the first step t where state(t+1) = state(t). Possible answers: "never: it enters a cycle of length λ", or "not decided within `max_steps`".
  3. **Does C occur:** fingerprint C and look it up. If the cycle closed without C appearing, the answer is a certain **never**.
  4. **Count of type X after N:** state after N, then count.
  5. **Does it enter a cycle:** the first repeated fingerprint gives the run-up length μ and the cycle length λ.
- **For random rules:** a repeated grid is not a true cycle, because the random numbers continue. Answers for random automata say so explicitly. This is a good detail to mention, and it links to the statistics planned for later.
- **Fallback:** if a cycle is too long to keep in memory, use Brent's algorithm (stretch).

---

## 8. Fixed randomness (same seed, same result)

- **One generator per run:** `numpy.random.Generator(PCG64(seed))`. Nothing uses Python's `random` or NumPy's global functions; a test checks this.
- **Fixed order of random draws:** the random initial world is drawn first. Then, each step, one full-grid array per `random()` call site, in rule order. Draws depend only on the step, never on timing or dictionary order.
- **The generator's internal state** (a small dictionary) is saved in every checkpoint, so a resumed run continues the same random sequence.
- **Shown three ways:**
  - **Test:** two runs with the same seed produce identical fingerprint sequences.
  - **Test:** different seeds produce different runs.
  - **Live:** two terminals side by side print the same fingerprints.

---

## 9. Saving, reloading and resuming

Each run gets its own folder, so it's self-contained and reproducible:
```
runs/forest_fire_brief/2026-11-03_14-05-12/
  automaton.json        copy of the spec used
  experiment.json       copy of the experiment, including the seed actually used
  checkpoint.npz        current grid, step, generator state, fingerprint list, spec digest
  generations.npz       stored generations (compressed; 2-state grids bit-packed), if enabled
  answers.json          answers to the questions
  run.log
```
- **Checkpoints:**
  - written every `checkpoint_every` steps and when the run is interrupted with Ctrl+C
  - written atomically (to a temporary file, then renamed), so a crash never leaves a broken checkpoint
- **Resume:** `cellab resume <run folder> [--steps K]` continues from the last checkpoint. It refuses if the automaton spec changed since (it compares the spec digest).
- **Key guarantee, tested:** run 100 steps in one go, versus run 40, stop, reload, run 60 more. The fingerprints and answers must be identical.
- **Extra:** start a new experiment from any stored generation (`"initial": {"type": "file", ...}`), e.g. "what happens from step 40 with a different seed".

---

## 10. Validation

Checked when a file is loaded, before anything runs. This is the constraint satisfaction topic from Lab 5.

- **Structure:** required fields, correct types, allowed values (`pydantic` models).
- **Meaning:**
  - every name used in a rule is a declared state, a parameter or a built-in
  - rule values fall within the declared states or range
  - the edge value is a valid state
  - the initial world uses only valid states and fits the grid
  - question parameters make sense (X is a state, C matches the grid size, N ≥ 0)
- **Messages name the file, the field and the problem.** For example: `forest_fire.json, rules[1].when: unknown name 'FLAME' (states are EMPTY, TREE, FIRE)`.
- **A folder of deliberately broken files** in the tests, each with the error it must produce.

---

## 11. Which automata

| Priority | Automata | Why |
|---|---|---|
| **Required (3)** | Rule 30, forest fire (deterministic + random), Brian's Brain | 1D and 2D, named states, randomness, and the brief's "harder automaton" |
| **Planned** | Sandpile, cyclic, voter, crystal growth, epidemic, diffusion, traffic (Rule 184) | Each is only a new file, which proves the engine is general |
| **Bonus** | Game of Life | Not in the brief's list; shows the system handles automata it wasn't built around |
| **Stretch** | Langton's ant | Needs the agent mode; at least show its file format |

Each automaton gets a spec file and an experiment file. Where the brief gives a starting grid, the experiment uses it.

---

## 12. Code layout and interfaces

```
cellab/
  spec.py           load + validate automaton specs
  experiment.py     load + validate experiment files
  expressions.py    safe rule expressions → NumPy
  neighbourhood.py  offsets, edges, padding
  engine.py         one step, the run loop
  randomness.py     seeded generator, saving/restoring its state
  fingerprints.py   state hashing, fingerprint table, cycle detection
  questions.py      the five questions
  storage.py        run folders, checkpoints, resume
  render.py         text, PNG and GIF output
  cli.py            `cellab check | run | resume | render`
automata/           spec files
experiments/        experiment files
tests/
```

Agree on these on day 1, so both of us can work in parallel:
```python
spec = load_spec(path)                      # -> Spec (validated)
exp = load_experiment(path)                 # -> Experiment (validated, spec loaded)
state = initial_state(spec, exp, rng)       # -> np.ndarray
new = step(spec, state, rng)                # -> np.ndarray, pure apart from rng draws
fp = fingerprint(state)                     # -> bytes (16)
result = run(spec, exp, run_dir)            # -> RunResult (fingerprints, answers, final state)
result = resume(run_dir, extra_steps)       # -> RunResult
```

**Command line:**
- `cellab check automata/forest_fire.json`: validate only.
- `cellab run experiments/forest_fire_brief.json`: run, answer, save.
- `cellab resume runs/.../<run>`: continue after a stop.
- `cellab render runs/.../<run> --gif`: images and an animation of the run.

**Dependencies:** numpy, pydantic, pillow, pytest, in a `requirements.txt` with pinned versions. Python 3.12 on both laptops.

---

## 13. Tests

**Results we already know** (computed while preparing Deliverable 1):

| Test | Expected |
|---|---|
| Rule 58, lab example, fixed 0 edges | `0001100` → `0011010` |
| Rule 30 on a ring, one live cell | width 8: run-up 1, cycle 40 · width 12: 24, 102 · width 16: 58, 6,016 |
| Rule 90, **any** start on a 16-cell ring | all cells 0 after at most 8 steps (a provable property) |
| Forest fire, brief grid | 8 trees and 5 fires after 2 steps; 2 trees after 3; stable from step 5; cycle of length 1 |
| Sandpile, brief grid `0 1 2 / 2 8 1 / 1 2 0` | stable after 3 steps at `1 3 2 / 0 2 3 / 3 0 1` |
| Brian's Brain, brief grid | the known generation 1 |

**Guarantees:**
- Same seed twice gives identical fingerprint sequences; different seeds give different ones.
- Interrupted and resumed equals uninterrupted (fingerprints and answers).
- Every broken input file is rejected with its expected message.
- **Generality guard:** the engine's source code contains none of the state names used in our example files (`TREE`, `FIRE`, `FIRING`…). A test that proves "nothing in the code is specific to one automaton", which also makes a nice slide.

---

## 14. Schedule for the week

Day 7 is the day before Lab 6. "A" and "B" are the two of us; the split is a suggestion.

| Day | Together | A: files and persistence | B: engine and answers |
|---|---|---|---|
| 1 | 2–3 h design session: freeze the spec and experiment formats (sections 4–5) and the interfaces (section 12); set up the package, venv, pytest | Write the 3 required automata as files | Neighbourhoods, edges, padding |
| 2 | – | Spec and experiment loading, validation, error messages | Expression compiler (`ast` whitelist → NumPy) |
| 3 | – | Run folders, checkpoints, Ctrl+C handling | `step()`, run loop, seeded generator |
| 4 | Integrate: the 3 required automata run end to end | Resume + resume-equivalence test | Fingerprints, cycle detection, the 5 questions + known-result tests |
| 5 | – | CLI, README usage section, broken-file tests | Determinism tests, generality guard, PNG/GIF rendering |
| 6 | Add the planned automata (files only) and their tests; fix what they reveal | | |
| 7 | Slides + demo rehearsal (section 15); push to GitHub | | |

If we're ahead: the stretch goals in section 16.
If we're behind: the required 3 automata and the full checklist in section 1 come before any extra automaton.

---

## 15. The Deliverable 2 presentation

About 8 slides plus a live demo (about 10 minutes), in the same style as Deliverable 1.

1. What Deliverable 2 asked, as a checklist with every point ticked
2. The two input files, with a real example
3. The rule language: one engine, many automata, and the generality-guard test
4. Fingerprints and the five questions
5. Saving and resuming, and the resume-equivalence test
6. Fixed randomness, and the same-seed test
7. Automata implemented, with GIFs
8. What's next (Deliverable 3) and the research preview, if done

**Demo moments** (short, visual, a little fun):
- **Stop and resume:** start a long run, press Ctrl+C halfway, resume. The final fingerprint matches the uninterrupted run shown earlier.
- **Twin runs:** two terminals, same seed, identical fingerprints scrolling in lockstep.
- **A new automaton in 30 seconds:** edit one number in the Brian's Brain file (`== 2` → `== 3`), rerun, and get completely different behaviour without touching code.
- **The forest from the brief burning** as a GIF, with the five answers printed under it.

---

## 16. Stretch goals (in order of value)

1. **Research preview:** with the finished engine, measure run-up and cycle lengths for all 256 elementary rules on rings of 8–16 cells, plotted by Wolfram class. It is the first result of the research part, and it feeds the Deliverable 3 dataset.
2. **Langton's ant** through a first version of the agent mode, using this format:
   ```json
   "agents": [{"name": "ant", "position": "center", "direction": "up"}],
   "agent_rules": [
     {"when": "cell == WHITE", "turn": "right", "set": "BLACK", "move": 1},
     {"when": "cell == BLACK", "turn": "left", "set": "WHITE", "move": 1}
   ]
   ```
3. **A first Streamlit page:** choose a file, run, watch the animation, read the answers. The start of the "interesting interface".
4. **Brent's cycle detection** for very long cycles.

---

## 17. Risks

| Risk | Plan |
|---|---|
| Designing the format takes too long | Timebox it to day 1, then freeze; extend later only by adding fields |
| The rule language gets complicated | Reuse Python's `ast` with a whitelist instead of a custom parser |
| Integration problems between the two halves | Agree on the interfaces in section 12 on day 1; integrate on day 4, not day 7 |
| Floating-point grids (diffusion) never repeat exactly | Rounding setting in the spec; the brief's example uses whole numbers |
| Random automata and cycles | Report "grid repeated" honestly instead of claiming a cycle (section 7) |
| Running out of time | The checklist in section 1 comes first; extra automata and stretch goals last |

---

## 18. Decisions to confirm before day 1

1. JSON for input files (alternative: YAML, which allows comments)
2. The expression-based rule language with Python's `ast` (section 4)
3. The required 3 automata (section 11)
4. Command line + GIFs for Deliverable 2; the graphical interface from Deliverable 3 on
5. The A/B split, and the exact date of Lab 6
