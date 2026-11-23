# Verification & Benchmark Report (`cuda`)

- **Models audited:** 51
- **PASS:** 63 | **WARN:** 4 | **FAIL:** 0 | **SKIP:** 0

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
| PASS | AnswerNet cycle length, classes 1-2 (test): error under half the median guess | MAE=0.05 vs median 7.02, MAE=2.40 vs median 9.18 |
| PASS | AnswerNet beats baselines (unseen, N=64) | model=0.7735 vs best_baseline=0.6566 (+0.1169) |
| PASS | Wolfram class ordering (unseen, N=64): Class 1 & 2 > Class 3 | C1=0.978, C2=0.833, C3=0.566 |
| PASS | Chaotic horizon collapse (unseen): Class 3 drops N=1 -> N=64 | C3 N=1=0.966 -> N=64=0.566 |
| PASS | AnswerNet cycle length, classes 1-2 (unseen): error under half the median guess | MAE=0.03 vs median 6.94, MAE=4.26 vs median 9.32 |
| PASS | StepNet + simulation: exact whole-run answers (test) | cycle length=1.0000, cycle start=1.0000, stable=1.0000, occurs=1.0000, cells N=64=1.0000 |
| PASS | StepNet + simulation: exact whole-run answers (unseen) | cycle length=1.0000, cycle start=1.0000, stable=1.0000, occurs=1.0000, cells N=64=1.0000 |

## Live Simulator Cross-Checks

| Status | Model | Target | Cells match | Extra |
|---|---|---|---|---|
| PASS | `step_elementary` | live engine rules [0, 30, 90, 110, 184], width=16, N=64 | 1.0000 | grids=1.0000 |
| PASS | `step_elementary` | live engine rules [0, 30, 90, 110, 184], width=32, N=64 | 1.0000 | grids=1.0000 |

## Cycle length by Wolfram class

Error is in steps; log error is |log(1 + said) - log(1 + true)|. The median guess is the training median.

| Method | Split | Class | Runs | Exactly right | Error | Median-guess error | Log error | Median-guess log error |
|---|---|---|---|---|---|---|---|---|
| AnswerNet (one pass) | test | all | 4360 | 0.455 | 42.40 | 43.77 | 0.221 | 1.116 |
| AnswerNet (one pass) | test | 1 uniform | 440 | 0.977 | 0.05 | 7.02 | 0.004 | 1.484 |
| AnswerNet (one pass) | test | 2 periodic | 3320 | 0.407 | 2.40 | 9.18 | 0.136 | 0.986 |
| AnswerNet (one pass) | test | 3 chaotic | 400 | 0.485 | 117.91 | 125.20 | 0.711 | 1.385 |
| AnswerNet (one pass) | test | 4 complex | 200 | 0.040 | 648.69 | 535.91 | 1.127 | 1.928 |
| AnswerNet (one pass) | unseen | all | 7600 | 0.180 | 494.46 | 498.83 | 0.971 | 1.575 |
| AnswerNet (one pass) | unseen | 1 uniform | 400 | 0.968 | 0.03 | 6.94 | 0.013 | 1.482 |
| AnswerNet (one pass) | unseen | 2 periodic | 5200 | 0.168 | 4.26 | 9.32 | 0.342 | 1.0 |
| AnswerNet (one pass) | unseen | 3 chaotic | 1200 | 0.076 | 3071.93 | 3074.53 | 3.903 | 4.187 |
| AnswerNet (one pass) | unseen | 4 complex | 800 | 0.021 | 61.76 | 63.05 | 1.146 | 1.445 |
| Gradient boosting | test | all | 4360 | 0.314 | 37.98 | 43.77 | 0.35 | 1.116 |
| Gradient boosting | test | 1 uniform | 440 | 0.823 | 0.46 | 7.02 | 0.103 | 1.484 |
| Gradient boosting | test | 2 periodic | 3320 | 0.294 | 2.97 | 9.18 | 0.235 | 0.986 |
| Gradient boosting | test | 3 chaotic | 400 | 0.072 | 124.20 | 125.21 | 1.103 | 1.385 |
| Gradient boosting | test | 4 complex | 200 | 0.010 | 529.28 | 535.91 | 1.297 | 1.928 |
| Gradient boosting | unseen | all | 7600 | 0.075 | 497.70 | 498.83 | 1.278 | 1.575 |
| Gradient boosting | unseen | 1 uniform | 400 | 0.417 | 0.81 | 6.95 | 0.3 | 1.482 |
| Gradient boosting | unseen | 2 periodic | 5200 | 0.073 | 8.23 | 9.32 | 0.71 | 1.0 |
| Gradient boosting | unseen | 3 chaotic | 1200 | 0.018 | 3072.74 | 3074.53 | 4.037 | 4.187 |
| Gradient boosting | unseen | 4 complex | 800 | 0.005 | 65.10 | 63.05 | 1.316 | 1.445 |
| StepNet + simulation | test | all | 4360 | 1.000 | 0.00 | 43.77 | 0.0 | 1.116 |
| StepNet + simulation | test | 1 uniform | 440 | 1.000 | 0.00 | 7.02 | 0.0 | 1.484 |
| StepNet + simulation | test | 2 periodic | 3320 | 1.000 | 0.00 | 9.18 | 0.0 | 0.986 |
| StepNet + simulation | test | 3 chaotic | 400 | 1.000 | 0.00 | 125.20 | 0.0 | 1.385 |
| StepNet + simulation | test | 4 complex | 200 | 1.000 | 0.00 | 535.91 | 0.0 | 1.928 |
| StepNet + simulation | unseen | all | 7600 | 1.000 | 0.00 | 498.83 | 0.0 | 1.575 |
| StepNet + simulation | unseen | 1 uniform | 400 | 1.000 | 0.00 | 6.94 | 0.0 | 1.482 |
| StepNet + simulation | unseen | 2 periodic | 5200 | 1.000 | 0.00 | 9.32 | 0.0 | 1.0 |
| StepNet + simulation | unseen | 3 chaotic | 1200 | 1.000 | 0.00 | 3074.53 | 0.0 | 4.187 |
| StepNet + simulation | unseen | 4 complex | 800 | 1.000 | 0.00 | 63.05 | 0.0 | 1.445 |

## Shared (5-head) vs Separate (1-head)

Mean ± standard deviation over the seeds trained (count in the last column). "within noise": the difference is smaller than the two standard deviations added.

| Split | Task | Metric | Shared | Single | Delta | Winner | Seeds |
|---|---|---|---|---|---|---|---|
| test | `state` | cells_correct (N=64) | 0.8900 ± 0.0021 | 0.9454 ± 0.0011 | -0.0554 | single | 3 / 3 |
| test | `count` | count_error (N=64) | 0.5329 ± 0.0079 | 0.4099 ± 0.0107 | +0.1229 | single | 3 / 3 |
| test | `stable` | stable F1 | 0.9920 ± 0.0014 | 0.9912 ± 0.0017 | +0.0008 | within noise | 3 / 3 |
| test | `cycle` | cycle_length MAE | 37.8385 ± 4.0179 | 34.9192 ± 0.7414 | +2.9193 | within noise | 3 / 3 |
| test | `occurs` | occurs F1 | 0.8579 ± 0.0025 | 0.8457 ± 0.0042 | +0.0122 | shared | 3 / 3 |
| unseen | `state` | cells_correct (N=64) | 0.7789 ± 0.0053 | 0.8008 ± 0.0035 | -0.0218 | single | 3 / 3 |
| unseen | `count` | count_error (N=64) | 1.2171 ± 0.0487 | 1.4072 ± 0.1546 | -0.1901 | within noise | 3 / 3 |
| unseen | `stable` | stable F1 | 0.6597 ± 0.0711 | 0.5632 ± 0.0292 | +0.0966 | within noise | 3 / 3 |
| unseen | `cycle` | cycle_length MAE | 496.4418 ± 2.3329 | 502.7369 ± 8.5250 | -6.2951 | within noise | 3 / 3 |
| unseen | `occurs` | occurs F1 | 0.7181 ± 0.0234 | 0.7359 ± 0.0042 | -0.0178 | within noise | 3 / 3 |

## Inference Speed Benchmarks

Not a like-for-like race: the simulator runs one grid at a time on the CPU, the networks run batches on the GPU, and AnswerNet's answers are not exact. Compare rows with the same batch and device.

| Method | Horizon N | Batch | ms / sample | Samples / s |
|---|---|---|---|---|
| cellab simulator (CPU) | 1 | 1 | 0.1074 | 9,306.7 |
| cellab simulator (CPU) | 8 | 1 | 0.8103 | 1,234.1 |
| cellab simulator (CPU) | 64 | 1 | 6.4859 | 154.2 |
| StepNet rollout (cuda) | 1 | 1 | 0.5348 | 1,869.9 |
| StepNet rollout (cuda) | 8 | 1 | 4.0753 | 245.4 |
| StepNet rollout (cuda) | 64 | 1 | 32.3296 | 30.9 |
| StepNet rollout (cuda) | 1 | 256 | 0.0021 | 467,727.5 |
| StepNet rollout (cuda) | 8 | 256 | 0.0167 | 59,857.1 |
| StepNet rollout (cuda) | 64 | 256 | 0.1314 | 7,609.0 |
| AnswerNet 1-pass (cuda) | 64 | 1 | 4.8192 | 207.5 |
| AnswerNet 1-pass (cuda) | 64 | 256 | 0.0188 | 53,267.0 |
| cellab 2D game_of_life_16 (CPU) | 32 | 1 | 9.7003 | 103.1 |
| StepNet 2D game_of_life_16 (cuda) | 32 | 64 | 0.2616 | 3,822.6 |
| AnswerNet 2D game_of_life_16 (cuda) | 32 | 64 | 0.0809 | 12,354.6 |

## Model Audit

| Status | Model | Epochs | Best ep | Time (s) | Cells N=1 | Cells last | vs Base | Notes |
|---|---|---|---|---|---|---|---|---|
| WARN | `answers_brians_brain_16` | 80 | 26 | 64.5 | 0.991 | 0.9278 | +0.0000 | last-horizon cells 0.928 no better than baseline 0.928 |
| PASS | `answers_crystal_growth_16` | 80 | 67 | 33.9 | 1.0 | 0.9931 | +0.5897 |  |
| PASS | `answers_cyclic_16` | 80 | 11 | 65.2 | 0.8822 | 0.8918 | +0.1127 |  |
| PASS | `answers_diffusion_16` | 80 | 68 | 21.6 | 0.4882 | 0.8749 | +0.5643 |  |
| PASS | `answers_ecosystem_16` | 80 | 71 | 21.6 | 0.835 | 0.6415 | +0.2966 |  |
| PASS | `answers_elementary` | 80 | 37 | 132.4 | 0.99 | 0.8877 | +0.2219 |  |
| PASS | `answers_elementary_on_elementary_w32_test` | None | 37 | None | 0.991 | 0.8853 | +0.2214 |  |
| PASS | `answers_elementary_seed1` | 80 | 37 | 131.3 | 0.9925 | 0.8919 | +0.2261 |  |
| PASS | `answers_elementary_seed2` | 80 | 38 | 130.4 | 0.9949 | 0.8903 | +0.2245 |  |
| PASS | `answers_epidemic_16` | 80 | 64 | 34.4 | 1.0 | 0.9998 | +0.0066 |  |
| PASS | `answers_falling_sand_16` | 80 | 67 | 17.0 | 0.9762 | 0.9746 | +0.2235 |  |
| PASS | `answers_forest_fire_16` | 80 | 49 | 33.8 | 0.9998 | 0.9933 | +0.0386 |  |
| PASS | `answers_forest_fire_regrowth_16` | 80 | 25 | 43.4 | 0.9785 | 0.8172 | +0.1846 |  |
| WARN | `answers_game_of_life_16` | 80 | 37 | 65.6 | 0.9994 | 0.8648 | +0.0000 | last-horizon cells 0.865 no better than baseline 0.865 |
| PASS | `answers_langtons_ant_16` | 80 | 69 | 32.1 | 0.9998 | 0.9674 | +0.0174 |  |
| WARN | `answers_predator_prey_16` | 80 | 32 | 21.6 | 0.6897 | 0.9671 | +0.0000 | last-horizon cells 0.967 no better than baseline 0.967 |
| PASS | `answers_sandpile_16` | 80 | 80 | 23.6 | 0.9986 | 0.7842 | +0.1977 |  |
| PASS | `answers_schelling_16` | 80 | 28 | 32.4 | 0.8027 | 0.9638 | +0.3503 |  |
| PASS | `answers_traffic_w16` | 80 | 52 | 36.1 | 0.9996 | 0.9992 | +0.1896 |  |
| PASS | `answers_voter_16` | 80 | 37 | 65.3 | 0.9988 | 0.9875 | +0.3578 |  |
| PASS | `gbdt_elementary` | None | None | None | None | None | — |  |
| PASS | `only_count_elementary` | 80 | 80 | 126.2 | 0.5809 | 0.6898 | +0.0240 |  |
| PASS | `only_count_elementary_seed1` | 80 | 75 | 128.5 | 0.482 | 0.4561 | -0.2097 |  |
| PASS | `only_count_elementary_seed2` | 80 | 75 | 123.5 | 0.4301 | 0.4437 | -0.2221 |  |
| PASS | `only_cycle_elementary` | 80 | 65 | 126.8 | 0.4952 | 0.5666 | -0.0992 |  |
| PASS | `only_cycle_elementary_seed1` | 80 | 69 | 126.8 | 0.444 | 0.406 | -0.2598 |  |
| PASS | `only_cycle_elementary_seed2` | 80 | 43 | 124.6 | 0.4552 | 0.4684 | -0.1974 |  |
| PASS | `only_occurs_elementary` | 80 | 20 | 126.5 | 0.5354 | 0.5811 | -0.0847 |  |
| PASS | `only_occurs_elementary_seed1` | 80 | 16 | 125.6 | 0.502 | 0.4747 | -0.1911 |  |
| PASS | `only_occurs_elementary_seed2` | 80 | 17 | 126.5 | 0.5018 | 0.5195 | -0.1463 |  |
| PASS | `only_stable_elementary` | 80 | 20 | 127.1 | 0.5346 | 0.5532 | -0.1126 |  |
| PASS | `only_stable_elementary_seed1` | 80 | 21 | 126.9 | 0.4659 | 0.4929 | -0.1729 |  |
| PASS | `only_stable_elementary_seed2` | 80 | 19 | 125.0 | 0.516 | 0.444 | -0.2218 |  |
| PASS | `only_state_elementary` | 80 | 58 | 123.7 | 1.0 | 0.9443 | +0.2785 |  |
| PASS | `only_state_elementary_seed1` | 80 | 60 | 123.4 | 1.0 | 0.9464 | +0.2806 |  |
| PASS | `only_state_elementary_seed2` | 80 | 57 | 121.2 | 1.0 | 0.9454 | +0.2796 |  |
| PASS | `simulated_elementary` | None | 39 | None | 1.0 | 1.0 | +0.3342 |  |
| PASS | `simulated_elementary_on_elementary_w32_test` | None | 39 | None | 1.0 | 1.0 | +0.3361 |  |
| PASS | `step_brians_brain_16` | 60 | 60 | 3.2 | 1.0 | 1.0 | +0.0722 |  |
| PASS | `step_crystal_growth_16` | 60 | 58 | 3.0 | 1.0 | 1.0 | +0.5966 |  |
| PASS | `step_cyclic_16` | 60 | 59 | 3.0 | 1.0 | 1.0 | +0.2210 |  |
| PASS | `step_diffusion_16` | 60 | 58 | 1.3 | 0.9984 | 0.9545 | +0.6438 |  |
| PASS | `step_elementary` | 40 | 39 | 10.4 | 1.0 | 1.0 | +0.3342 |  |
| PASS | `step_elementary_on_elementary_w32_test` | None | 39 | None | 1.0 | 1.0 | +0.3361 |  |
| PASS | `step_epidemic_16` | 60 | 59 | 3.1 | 1.0 | 1.0 | +0.0069 |  |
| WARN | `step_falling_sand_16` | 60 | 56 | 1.8 | 0.9785 | 0.7508 | -0.0002 | 1-step cell accuracy 0.979 < 0.99 |
| PASS | `step_forest_fire_16` | 60 | 59 | 3.0 | 1.0 | 1.0 | +0.0453 |  |
| PASS | `step_game_of_life_16` | 60 | 58 | 3.1 | 1.0 | 1.0 | +0.1353 |  |
| PASS | `step_sandpile_16` | 60 | 58 | 2.4 | 1.0 | 0.8853 | +0.2987 |  |
| PASS | `step_traffic_w16` | 60 | 52 | 3.6 | 1.0 | 1.0 | +0.1904 |  |
| PASS | `step_voter_16` | 60 | 57 | 3.1 | 1.0 | 1.0 | +0.3703 |  |
