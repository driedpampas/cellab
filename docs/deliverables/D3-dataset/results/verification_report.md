# Verification & Benchmark Report (`cuda (L4) + local CPU fixes`)

- **Models audited:** 36
- **PASS:** 49 | **WARN:** 1 | **FAIL:** 0 | **SKIP:** 0

## Core Hypothesis Checks

| Status | Check | Detail |
|---|---|---|
| PASS | StepNet 16-cell exactness (test, N=64) | cells=1.0000, grids=1.0000 |
| PASS | StepNet 16-cell exactness (unseen, N=64) | cells=1.0000, grids=1.0000 |
| PASS | StepNet 32-cell transfer exactness (test, N=64) | cells=1.0000, grids=1.0000 |
| PASS | StepNet 32-cell transfer exactness (unseen, N=64) | cells=1.0000, grids=1.0000 |
| PASS | AnswerNet beats baselines (test, N=64) | model=0.8877 vs best_baseline=0.6658 (+0.2219) |
| PASS | Wolfram class ordering (test, N=64): Class 1 & 2 > Class 3 | C1=0.997, C2=0.902, C3=0.810 |
| PASS | Chaotic horizon collapse (test): Class 3 drops N=1 -> N=64 | C3 N=1=0.977 -> N=64=0.810 |
| PASS | AnswerNet cycle length beats median (test) | MAE=42.40 vs common=43.77 |
| PASS | AnswerNet beats baselines (unseen, N=64) | model=0.7735 vs best_baseline=0.6566 (+0.1169) |
| PASS | Wolfram class ordering (unseen, N=64): Class 1 & 2 > Class 3 | C1=0.978, C2=0.833, C3=0.566 |
| PASS | Chaotic horizon collapse (unseen): Class 3 drops N=1 -> N=64 | C3 N=1=0.966 -> N=64=0.566 |
| PASS | AnswerNet cycle length beats median (unseen) | MAE=494.46 vs common=498.83 |

## Live Simulator Cross-Checks

| Status | Model | Target | Cells match | Extra |
|---|---|---|---|---|
| PASS | `step_elementary` | live engine rules [0, 30, 90, 110, 184], width=16, N=64 | 1.0000 | grids=1.0000 |
| PASS | `step_elementary` | live engine rules [0, 30, 90, 110, 184], width=32, N=64 | 1.0000 | grids=1.0000 |

## Shared (5-head) vs Separate (1-head)

| Split | Task | Metric | Shared | Single | Delta | Winner |
|---|---|---|---|---|---|---|
| test | `state` | cells_correct (N=64) | 0.8877 | 0.9450 | -0.0573 | single |
| test | `count` | count_error (N=64) | 0.5419 | 0.3953 | +0.1465 | single |
| test | `stable` | stable F1 | 0.9935 | 0.9911 | +0.0024 | shared |
| test | `cycle` | cycle_length MAE | 42.4037 | 31.4328 | +10.9709 | single |
| test | `occurs` | occurs F1 | 0.8551 | 0.8401 | +0.0149 | shared |
| unseen | `state` | cells_correct (N=64) | 0.7735 | 0.8019 | -0.0285 | single |
| unseen | `count` | count_error (N=64) | 1.1886 | 1.4176 | -0.2290 | shared |
| unseen | `stable` | stable F1 | 0.6003 | 0.5176 | +0.0827 | shared |
| unseen | `cycle` | cycle_length MAE | 494.4568 | 511.6893 | -17.2325 | shared |
| unseen | `occurs` | occurs F1 | 0.7414 | 0.7400 | +0.0014 | shared |

## Inference Speed Benchmarks

| Method | Horizon N | Batch | ms / sample | Samples / s |
|---|---|---|---|---|
| cellab simulator (CPU) | 1 | 1 | 0.1080 | 9,259.4 |
| cellab simulator (CPU) | 8 | 1 | 0.8141 | 1,228.3 |
| cellab simulator (CPU) | 64 | 1 | 6.4702 | 154.6 |
| StepNet rollout (cuda) | 1 | 1 | 0.5403 | 1,850.8 |
| StepNet rollout (cuda) | 8 | 1 | 4.0996 | 243.9 |
| StepNet rollout (cuda) | 64 | 1 | 32.1818 | 31.1 |
| StepNet rollout (cuda) | 1 | 256 | 0.0021 | 473,803.5 |
| StepNet rollout (cuda) | 8 | 256 | 0.0170 | 58,664.2 |
| StepNet rollout (cuda) | 64 | 256 | 0.1303 | 7,676.5 |
| AnswerNet 1-pass (cuda) | 64 | 1 | 4.8526 | 206.1 |
| AnswerNet 1-pass (cuda) | 64 | 256 | 0.0184 | 54,365.4 |
| cellab 2D game_of_life_16 (CPU) | 32 | 1 | 9.7103 | 103.0 |
| StepNet 2D game_of_life_16 (cuda) | 32 | 64 | 0.2639 | 3,789.8 |
| AnswerNet 2D game_of_life_16 (cuda) | 32 | 64 | 0.0799 | 12,520.0 |

## Model Audit

| Status | Model | Epochs | Best ep | Time (s) | Cells N=1 | Cells last | vs Base | Notes |
|---|---|---|---|---|---|---|---|---|
| PASS | `answers_brians_brain_16` | None | 16 | None | 0.9055 | 0.9278 | +0.0000 |  |
| PASS | `answers_crystal_growth_16` | None | 59 | None | 1.0 | 0.9907 | +0.5873 |  |
| PASS | `answers_cyclic_16` | None | 16 | None | 0.9145 | 0.9061 | +0.1271 |  |
| PASS | `answers_ecosystem_16` | None | 79 | None | 0.8361 | 0.6448 | +0.2998 |  |
| PASS | `answers_elementary` | None | 37 | None | 0.99 | 0.8877 | +0.2219 |  |
| PASS | `answers_elementary_on_elementary_w32_test` | None | 37 | None | 0.991 | 0.8853 | +0.2214 |  |
| PASS | `answers_epidemic_16` | None | 64 | None | 0.9998 | 0.9996 | +0.0064 |  |
| PASS | `answers_falling_sand_16` | None | 69 | None | 0.9757 | 0.9736 | +0.2226 |  |
| PASS | `answers_forest_fire_16` | None | 60 | None | 0.9998 | 0.9925 | +0.0379 |  |
| PASS | `answers_forest_fire_regrowth_16` | None | 31 | None | 0.9755 | 0.8204 | +0.1878 |  |
| PASS | `answers_game_of_life_16` | None | 33 | None | 0.996 | 0.8647 | -0.0000 |  |
| PASS | `answers_langtons_ant_16` | None | 80 | None | 0.9999 | 0.9676 | +0.0176 |  |
| PASS | `answers_predator_prey_16` | None | 43 | None | 0.6909 | 0.9671 | +0.0000 |  |
| PASS | `answers_sandpile_16` | None | 80 | None | 0.9988 | 0.7847 | +0.1982 |  |
| PASS | `answers_schelling_16` | None | 45 | None | 0.8068 | 0.9647 | +0.3512 |  |
| PASS | `answers_traffic_w16` | None | 52 | None | 1.0 | 0.9996 | +0.1900 |  |
| PASS | `answers_voter_16` | None | 39 | None | 0.9995 | 0.9884 | +0.3588 |  |
| PASS | `gbdt_elementary` | None | None | None | None | None | — |  |
| PASS | `only_count_elementary` | None | 77 | None | 0.6387 | 0.7499 | +0.0841 |  |
| PASS | `only_cycle_elementary` | None | 57 | None | 0.5079 | 0.5252 | -0.1406 |  |
| PASS | `only_occurs_elementary` | None | 12 | None | 0.5129 | 0.5382 | -0.1276 |  |
| PASS | `only_stable_elementary` | None | 23 | None | 0.5031 | 0.5298 | -0.1360 |  |
| PASS | `only_state_elementary` | None | 58 | None | 1.0 | 0.945 | +0.2792 |  |
| PASS | `step_brians_brain_16` | None | 60 | None | 1.0 | 1.0 | +0.0722 |  |
| PASS | `step_crystal_growth_16` | None | 25 | None | 1.0 | 1.0 | +0.5966 |  |
| PASS | `step_cyclic_16` | None | 58 | None | 1.0 | 1.0 | +0.2210 |  |
| PASS | `step_diffusion_16` | None | 50 | None | 0.9975 | 0.8049 | +0.4943 |  |
| PASS | `step_elementary` | None | 37 | None | 1.0 | 1.0 | +0.3342 |  |
| PASS | `step_elementary_on_elementary_w32_test` | None | 37 | None | 1.0 | 1.0 | +0.3361 |  |
| PASS | `step_epidemic_16` | None | 25 | None | 1.0 | 1.0 | +0.0069 |  |
| WARN | `step_falling_sand_16` | None | 56 | None | 0.9791 | 0.7497 | -0.0014 | 1-step cell accuracy 0.979 < 0.99 |
| PASS | `step_forest_fire_16` | None | 59 | None | 1.0 | 1.0 | +0.0453 |  |
| PASS | `step_game_of_life_16` | None | 58 | None | 1.0 | 1.0 | +0.1353 |  |
| PASS | `step_sandpile_16` | None | 59 | None | 1.0 | 0.6586 | +0.0721 |  |
| PASS | `step_traffic_w16` | None | 54 | None | 1.0 | 1.0 | +0.1904 |  |
| PASS | `step_voter_16` | None | 58 | None | 1.0 | 1.0 | +0.3703 |  |
