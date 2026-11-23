# Verification & Benchmark Report (`cuda`)

- **Models audited:** 34
- **PASS:** 47 | **WARN:** 1 | **FAIL:** 0 | **SKIP:** 0

## Core Hypothesis Checks

| Status | Check | Detail |
|---|---|---|
| PASS | StepNet 16-cell exactness (test, N=64) | cells=1.0000, grids=1.0000 |
| PASS | StepNet 16-cell exactness (unseen, N=64) | cells=1.0000, grids=1.0000 |
| PASS | StepNet 32-cell transfer exactness (test, N=64) | cells=1.0000, grids=1.0000 |
| PASS | StepNet 32-cell transfer exactness (unseen, N=64) | cells=1.0000, grids=1.0000 |
| PASS | AnswerNet beats baselines (test, N=64) | model=0.8829 vs best_baseline=0.6658 (+0.2171) |
| PASS | Wolfram class ordering (test, N=64): Class 1 & 2 > Class 3 | C1=1.000, C2=0.895, C3=0.817 |
| PASS | Chaotic horizon collapse (test): Class 3 drops N=1 -> N=64 | C3 N=1=0.933 -> N=64=0.817 |
| PASS | AnswerNet cycle length beats median (test) | MAE=33.44 vs common=43.77 |
| PASS | AnswerNet beats baselines (unseen, N=64) | model=0.7621 vs best_baseline=0.6566 (+0.1055) |
| PASS | Wolfram class ordering (unseen, N=64): Class 1 & 2 > Class 3 | C1=0.974, C2=0.815, C3=0.567 |
| PASS | Chaotic horizon collapse (unseen): Class 3 drops N=1 -> N=64 | C3 N=1=0.903 -> N=64=0.567 |
| PASS | AnswerNet cycle length beats median (unseen) | MAE=494.04 vs common=498.83 |

## Live Simulator Cross-Checks

| Status | Model | Target | Cells match | Extra |
|---|---|---|---|---|
| PASS | `step_elementary` | live engine rules [0, 30, 90, 110, 184], width=16, N=64 | 1.0000 | grids=1.0000 |
| PASS | `step_elementary` | live engine rules [0, 30, 90, 110, 184], width=32, N=64 | 1.0000 | grids=1.0000 |

## Shared (5-head) vs Separate (1-head)

| Split | Task | Metric | Shared | Single | Delta | Winner |
|---|---|---|---|---|---|---|
| test | `state` | cells_correct (N=64) | 0.8829 | 0.9450 | -0.0621 | single |
| test | `count` | count_error (N=64) | 0.5259 | 0.3953 | +0.1306 | single |
| test | `stable` | stable F1 | 0.9897 | 0.9911 | -0.0013 | single |
| test | `cycle` | cycle_length MAE | 33.4413 | 31.4328 | +2.0085 | single |
| test | `occurs` | occurs F1 | 0.8493 | 0.8401 | +0.0092 | shared |
| unseen | `state` | cells_correct (N=64) | 0.7621 | 0.8019 | -0.0399 | single |
| unseen | `count` | count_error (N=64) | 1.2757 | 1.4176 | -0.1419 | shared |
| unseen | `stable` | stable F1 | 0.7115 | 0.5176 | +0.1939 | shared |
| unseen | `cycle` | cycle_length MAE | 494.0425 | 511.6893 | -17.6469 | shared |
| unseen | `occurs` | occurs F1 | 0.7038 | 0.7400 | -0.0362 | single |

## Inference Speed Benchmarks

| Method | Horizon N | Batch | ms / sample | Samples / s |
|---|---|---|---|---|
| cellab simulator (CPU) | 1 | 1 | 0.1097 | 9,115.5 |
| cellab simulator (CPU) | 8 | 1 | 0.8216 | 1,217.2 |
| cellab simulator (CPU) | 64 | 1 | 6.4917 | 154.0 |
| StepNet rollout (cuda) | 1 | 1 | 0.5062 | 1,975.6 |
| StepNet rollout (cuda) | 8 | 1 | 3.8685 | 258.5 |
| StepNet rollout (cuda) | 64 | 1 | 30.8064 | 32.5 |
| StepNet rollout (cuda) | 1 | 256 | 0.0020 | 493,986.5 |
| StepNet rollout (cuda) | 8 | 256 | 0.0159 | 62,686.7 |
| StepNet rollout (cuda) | 64 | 256 | 0.1237 | 8,086.2 |
| AnswerNet 1-pass (cuda) | 64 | 1 | 4.6419 | 215.4 |
| AnswerNet 1-pass (cuda) | 64 | 256 | 0.0178 | 56,216.5 |
| cellab 2D game_of_life_16 (CPU) | 32 | 1 | 9.5228 | 105.0 |
| StepNet 2D game_of_life_16 (cuda) | 32 | 64 | 0.2490 | 4,015.6 |
| AnswerNet 2D game_of_life_16 (cuda) | 32 | 64 | 0.0765 | 13,068.4 |

## Model Audit

| Status | Model | Epochs | Best ep | Time (s) | Cells N=1 | Cells last | vs Base | Notes |
|---|---|---|---|---|---|---|---|---|
| PASS | `answers_brians_brain_16` | 80 | 16 | 66.5 | 0.9055 | 0.9278 | +0.0000 |  |
| PASS | `answers_crystal_growth_16` | 80 | 59 | 34.7 | 1.0 | 0.9907 | +0.5873 |  |
| PASS | `answers_cyclic_16` | 80 | 16 | 65.6 | 0.9145 | 0.9061 | +0.1271 |  |
| PASS | `answers_ecosystem_16` | 80 | 79 | 21.6 | 0.8361 | 0.6448 | +0.2998 |  |
| PASS | `answers_elementary` | 80 | 43 | 132.5 | 0.9751 | 0.8829 | +0.2171 |  |
| PASS | `answers_elementary_on_elementary_w32_test` | None | 43 | None | 0.9763 | 0.8779 | +0.2139 |  |
| PASS | `answers_epidemic_16` | 80 | 64 | 34.6 | 0.9998 | 0.9996 | +0.0064 |  |
| PASS | `answers_falling_sand_16` | 80 | 69 | 17.5 | 0.9757 | 0.9736 | +0.2226 |  |
| PASS | `answers_forest_fire_16` | 80 | 60 | 34.9 | 0.9998 | 0.9925 | +0.0379 |  |
| PASS | `answers_forest_fire_regrowth_16` | 80 | 31 | 43.2 | 0.9755 | 0.8204 | +0.1878 |  |
| PASS | `answers_game_of_life_16` | 80 | 33 | 65.8 | 0.996 | 0.8647 | -0.0000 |  |
| PASS | `answers_langtons_ant_16` | 80 | 80 | 32.4 | 0.9999 | 0.9676 | +0.0176 |  |
| PASS | `answers_predator_prey_16` | 80 | 43 | 21.6 | 0.6909 | 0.9671 | +0.0000 |  |
| PASS | `answers_sandpile_16` | 80 | 80 | 24.1 | 0.9988 | 0.7847 | +0.1982 |  |
| PASS | `answers_schelling_16` | 80 | 45 | 32.4 | 0.8068 | 0.9647 | +0.3512 |  |
| PASS | `answers_traffic_w16` | 80 | 52 | 36.7 | 1.0 | 0.9996 | +0.1900 |  |
| PASS | `answers_voter_16` | 80 | 39 | 66.3 | 0.9995 | 0.9884 | +0.3588 |  |
| PASS | `only_count_elementary` | 80 | 77 | 126.6 | 0.6387 | 0.7499 | +0.0841 |  |
| PASS | `only_cycle_elementary` | 80 | 57 | 126.5 | 0.5079 | 0.5252 | -0.1406 |  |
| PASS | `only_occurs_elementary` | 80 | 12 | 126.1 | 0.5129 | 0.5382 | -0.1276 |  |
| PASS | `only_stable_elementary` | 80 | 23 | 126.3 | 0.5031 | 0.5298 | -0.1360 |  |
| PASS | `only_state_elementary` | 80 | 58 | 123.5 | 1.0 | 0.945 | +0.2792 |  |
| PASS | `step_brians_brain_16` | 60 | 60 | 3.2 | 1.0 | 1.0 | +0.0722 |  |
| PASS | `step_crystal_growth_16` | 60 | 60 | 3.0 | 1.0 | 0.9077 | +0.5043 |  |
| PASS | `step_cyclic_16` | 60 | 58 | 3.2 | 1.0 | 1.0 | +0.2210 |  |
| PASS | `step_elementary` | 40 | 37 | 11.8 | 1.0 | 1.0 | +0.3342 |  |
| PASS | `step_elementary_on_elementary_w32_test` | None | 37 | None | 1.0 | 1.0 | +0.3361 |  |
| PASS | `step_epidemic_16` | 60 | 59 | 3.1 | 1.0 | 0.0248 | -0.9684 |  |
| WARN | `step_falling_sand_16` | 60 | 56 | 1.8 | 0.9791 | 0.7497 | -0.0014 | 1-step cell accuracy 0.979 < 0.99 |
| PASS | `step_forest_fire_16` | 60 | 59 | 3.0 | 1.0 | 1.0 | +0.0453 |  |
| PASS | `step_game_of_life_16` | 60 | 58 | 3.2 | 1.0 | 1.0 | +0.1353 |  |
| PASS | `step_sandpile_16` | 60 | 59 | 2.4 | 1.0 | 0.6586 | +0.0721 |  |
| PASS | `step_traffic_w16` | 60 | 54 | 3.4 | 1.0 | 1.0 | +0.1904 |  |
| PASS | `step_voter_16` | 60 | 58 | 3.1 | 1.0 | 1.0 | +0.3703 |  |
