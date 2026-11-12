# The datasets

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


| Dataset | Automaton | Samples | Grid | Horizons | Rules | Stable | Cycle found | Median cycle | C occurs: yes / no / unknown | File |
|---|---|---|---|---|---|---|---|---|---|---|
| `brians_brain_16` | Brian's Brain | 3,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 34% | 100% | 16 | 50% / 50% / 0% | 1.1 MB |
| `crystal_growth_16` | Crystal growth | 3,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 100% | 100% | 1 | 50% / 50% / 0% | 0.6 MB |
| `cyclic_16` | Cyclic automaton | 3,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 63% | 100% | 1 | 50% / 50% / 0% | 2.2 MB |
| `diffusion_16` | Diffusion | 1,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 100% | 100% | 1 | 100% / 0% / 0% | 2.9 MB |
| `ecosystem_16` | Artificial ecosystem | 1,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | random | 2% | 0% | – | 50% / 0% / 50% | 0.7 MB |
| `elementary_w16` | Elementary rule | 51,200 | 16 | 1, 2, 4, 8, 16, 32, 64 | exact | 32% | 100% | 14 | 52% / 48% / 0% | 1.9 MB |
| `elementary_w32_test` | Elementary rule | 7,680 | 32 | 1, 2, 4, 8, 16, 32, 64 | exact | 30% | 84% | 3 | 51% / 41% / 8% | 0.6 MB |
| `epidemic_16` | Epidemic | 3,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 100% | 100% | 1 | 50% / 50% / 0% | 0.9 MB |
| `falling_sand_16` | Falling sand | 1,500 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 34% | 100% | 2 | 50% / 50% / 0% | 0.7 MB |
| `forest_fire_16` | Forest fire | 3,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 100% | 100% | 1 | 50% / 50% / 0% | 0.9 MB |
| `forest_fire_regrowth_16` | Forest fire with regrowth | 2,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | random | 52% | 0% | – | 50% / 0% / 50% | 0.7 MB |
| `game_of_life_16` | Game of Life | 3,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 60% | 100% | 1 | 50% / 50% / 0% | 0.9 MB |
| `langtons_ant_16` | Langton's ant | 1,500 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 0% | 0% | – | 50% / 0% / 50% | 0.3 MB |
| `predator_prey_16` | Predator-prey | 1,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | random | 34% | 0% | – | 50% / 0% / 50% | 0.4 MB |
| `sandpile_16` | Sandpile | 2,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 100% | 100% | 1 | 100% / 0% / 0% | 1.7 MB |
| `schelling_16` | Schelling segregation | 1,500 | 16 x 16 | 1, 2, 4, 8, 16, 32 | random | 92% | 0% | – | 50% / 0% / 50% | 0.6 MB |
| `traffic_w16` | Traffic | 3,000 | 16 | 1, 2, 4, 8, 16, 32 | exact | 1% | 100% | 16 | 52% / 48% / 0% | 0.1 MB |
| `voter_16` | Voter model | 3,000 | 16 x 16 | 1, 2, 4, 8, 16, 32 | exact | 100% | 100% | 1 | 50% / 50% / 0% | 0.5 MB |

94,380 samples in all.

- **`brians_brain_16`:** 16 x 16 grid, random starts of several densities. 
- **`crystal_growth_16`:** 16 x 16 grid, random starts of several densities. A few seed crystals on an empty grid.
- **`cyclic_16`:** 16 x 16 grid, random starts of several densities. Four levels, threshold 2.
- **`diffusion_16`:** 16 x 16 grid, random starts of several densities. Decimal values; a single hot cell or random heat.
- **`ecosystem_16`:** 16 x 16 grid, random starts of several densities. Random moves and growth: each sample is one seeded realisation.
- **`elementary_w16`:** All 256 elementary rules on a ring of 16 cells: 200 runs per rule from random rows of different densities. With 16 cells there are 65,536 possible rows, so every run is followed until its cycle is found and all five questions have exact answers.
- **`elementary_w32_test`:** A test set on a wider ring (32 cells) for the question 'does a model trained on 16 cells work on other sizes?'. With 2^32 possible rows the cycles of chaotic rules cannot be found, so only the answers after N steps are complete.
- **`epidemic_16`:** 16 x 16 grid, random starts of several densities. 
- **`falling_sand_16`:** 16 x 16 grid, random starts of several densities. Deterministic: sand settles; water keeps moving in a short cycle.
- **`forest_fire_16`:** 16 x 16 grid, random starts of several densities. The deterministic forest fire of the brief; every run burns out, so all answers are exact.
- **`forest_fire_regrowth_16`:** 16 x 16 grid, random starts of several densities. Random rules (regrowth and lightning): each sample is one seeded realisation, so targets are noisy by nature.
- **`game_of_life_16`:** 16 x 16 grid, random starts of several densities. Runs are followed for up to 1,000 steps to find their cycle.
- **`langtons_ant_16`:** 16 x 16 grid, random starts of several densities. One ant in the centre, heading north, on a random or a white grid. 16 x 16 cells with an ant have far too many states to find a cycle, so cycle answers are mostly 'unknown'.
- **`predator_prey_16`:** 16 x 16 grid, random starts of several densities. Random moves: each sample is one seeded realisation.
- **`sandpile_16`:** 16 x 16 grid, random starts of several densities. Grains are whole numbers without an upper limit, so there are no per-value counts.
- **`schelling_16`:** 16 x 16 grid, random starts of several densities. Random moves: each sample is one seeded realisation.
- **`traffic_w16`:** A ring road of 16 cells with random cars of several densities.
- **`voter_16`:** 16 x 16 grid, random starts of several densities. 
