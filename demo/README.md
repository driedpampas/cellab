# Live demo for Deliverable 2

Five short steps, run from the project folder in PowerShell with the virtual environment active
(`.venv\Scripts\Activate.ps1`). Each takes under a minute. The expected output is given under each step,
so you can tell at a glance that it worked.

Before starting, remove the folders of an earlier rehearsal:

```powershell
Remove-Item -Recurse -Force runs\live -ErrorAction SilentlyContinue
```

## 1. Mistakes in input files are reported clearly

```powershell
cellab check automata/forest_fire.json demo/fire_typo.json demo/fire_spelling.json demo/fire_attack.json
```

- `fire_typo.json` uses `FIER`: the message says `did you mean 'FIRE'?`
- `fire_spelling.json` writes `neighborhood`: the message lists the allowed fields
- `fire_attack.json` tries to run a shell command from a rule: it is rejected, because rules are never run as Python

## 2. The brief's forest fire

```powershell
cellab run experiments/forest_fire_brief.json --show --runs-dir runs/live
```

Prints every generation and the answers: stable from step 5, 2 trees after 3 steps.

## 3. Stop, resume, and compare with an uninterrupted run

```powershell
cellab run experiments/forest_fire_regrowth.json --seed 5 --out runs/live/straight
```

```powershell
cellab run experiments/forest_fire_regrowth.json --seed 5 --out runs/live/stopped --stop-at 120
```

```powershell
cellab resume runs/live/stopped
```

```powershell
fc.exe /b runs\live\straight\generations.bin runs\live\stopped\generations.bin
```

Expected: `FC: no differences encountered`. In PowerShell write `fc.exe`, because `fc` alone means something else.
Pressing Ctrl+C during the first run works the same way as `--stop-at`.

## 4. Same seed, same result

```powershell
cellab run demo/twin.json --seed 42 --trace --runs-dir runs/live
```

Run it twice: both print the same fingerprints (step 0 is `46f060264b4732cb`, step 8 is `48e711909969d86c`).
Then run it with `--seed 43`: the fingerprints differ from step 0.

## 5. Change a rule in the file, not in the code

```powershell
cellab run experiments/brians_brain_random.json --runs-dir runs/live
```

Expected: never stable, a cycle of length 100 from step 750, 288 firing cells after 100 steps.

Now open `automata/brians_brain.json` and change `count(FIRING) == 2` to `count(FIRING) == 3`. Run the same
command again. Expected: stable from step 6, 0 firing cells after 100 steps. The activity dies out almost at once.

Put the file back afterwards:

```powershell
git restore automata/brians_brain.json
```
