# Run bluesky_jepa_set

## Layout and setup (once)

    <repo>/                    public competition repo, branch AI4REAL-NET-Competition, commit 0093001
    <repo>/bluesky_jepa_set/   this folder (unzip it here)

    git clone -b AI4REAL-NET-Competition https://github.com/TUDelft-CNS-ATM/bluesky-gym.git bluesky-gym
    cd bluesky-gym && uv sync && uv add torch     # skip "uv add torch" if torch is in the lock file
    unzip bluesky_jepa_set.zip                    # creates bluesky_jepa_set/
    cd bluesky_jepa_set

Run every command from `bluesky_jepa_set/` with `PYTHONPATH=..`. No path in the code is absolute.
Packages used: torch, numpy, gymnasium, pettingzoo, Pillow (GIF). All are in the base repo environment. pytest is optional and is not in the base repo.

Device: `--device cpu` is the default. Use `--device cuda` only if the GPU is in the list from:

    ../.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"

Workers: `scripts.run_grid --workers 6` starts 6 separate processes with one torch thread each. BlueSky allows one environment per process.

## Tests (about 2 seconds, no BlueSky)

    PYTHONPATH=.. ../.venv/bin/python -m tests.run_tests
    PYTHONPATH=.. ../.venv/bin/python -m pytest tests -q      # only if pytest is installed

## Check before the first real run: same seed, bit-identical output (about 1 minute)

    PYTHONPATH=.. ../.venv/bin/python -m scripts.check_determinism

Expected last line: `BIT-IDENTICAL: yes`. Output goes to `determinism/`.

## Smoke runs (timing; 3000 RL steps per run, full pipeline)

    PYTHONPATH=.. ~/bin/notify_run.sh grid_single_smoke ../.venv/bin/python -m scripts.run_grid --setting single --seeds 0 1 --steps 3000 --tag _smoke
    PYTHONPATH=.. ~/bin/notify_run.sh grid_multi_smoke  ../.venv/bin/python -m scripts.run_grid --setting multi  --seeds 0 1 --steps 3000 --tag _smoke

Read `seconds_per_step` in `results/compare_smoke.md` (column s/step). Real run time per job is about `30000 * s/step`. 12 jobs with 6 workers run in 2 waves.

## Overfit runs (3 seeds, cells R A B C, 30 000 steps each)

Single agent first:

    PYTHONPATH=.. ~/bin/notify_run.sh grid_single ../.venv/bin/python -m scripts.run_grid --setting single --cells R A B C --seeds 0 1 2 --steps 30000 --workers 6 --device cpu

Then multi agent (2 aircraft):

    PYTHONPATH=.. ~/bin/notify_run.sh grid_multi ../.venv/bin/python -m scripts.run_grid --setting multi --n-agents 2 --cells R A B C --seeds 0 1 2 --steps 30000 --workers 6 --device cpu

One cell only: replace `--cells R A B C` with for example `--cells A`. Pass extra training flags after `--train-args`, as the last option (for example `--train-args --shaping off`).

The grid does these stages:
0. `scripts.collect`: 6000 transitions from the fixed scenario to `data/transitions_SETTING.npz` (skipped if the file exists).
1. `scripts.imitate` for each encoder. Stop rule: no RL run for an encoder that fails.
2. For each cell and seed: `scripts.pretrain_jepa` (cells B and C; stop rule: no RL run after a collapse), then `scripts.train`.
3. `scripts.compare`.

The same jobs as single commands (example: cell C, seed 0, multi agent):

    PYTHONPATH=.. ../.venv/bin/python -m scripts.collect --setting multi --n-agents 2 --transitions 6000
    PYTHONPATH=.. ~/bin/notify_run.sh imit_ma_set_tf ../.venv/bin/python -m scripts.imitate --name imit_ma_set_tf --data data/transitions_multi.npz --setting multi --encoder set_tf
    PYTHONPATH=.. ~/bin/notify_run.sh jepa_ma_C_s0 ../.venv/bin/python -m scripts.pretrain_jepa --name jepa_ma_C_s0 --data data/transitions_multi.npz --encoder set_tf --seed 0
    PYTHONPATH=.. ~/bin/notify_run.sh ma_C_s0 ../.venv/bin/python -m scripts.train --name ma_C_s0 --setting multi --n-agents 2 --encoder set_tf --jepa on --jepa-ckpt checkpoints/jepa_ma_C_s0.pt --seed 0 --steps 30000

## Evaluation: metrics table and GIF (about 1 minute each, no training)

    PYTHONPATH=.. ../.venv/bin/python -m scripts.evaluate --model models/sa_A_s0.pt --gif gifs/sa_A_s0.gif
    PYTHONPATH=.. ../.venv/bin/python -m scripts.evaluate --model models/ma_A_s0.pt --gif gifs/ma_A_s0.gif

Baselines without a model, for the same table and GIF:

    PYTHONPATH=.. ../.venv/bin/python -m scripts.evaluate --policy zero --setting single --gif gifs/zero_sa.gif
    PYTHONPATH=.. ../.venv/bin/python -m scripts.evaluate --policy controller --setting multi --gif gifs/controller_ma.gif

Options: `--episodes N` uses the scenario seeds reset-seed, reset-seed + 1, ...; `--n-agents N` flies a model with a different aircraft count; `--stochastic` samples actions. The official score stays `python -m scripts.evaluate_competition` in the repo root.

## Output files

| File | Content |
|---|---|
| `results/NAME_summary.json` | status EXPLORATORY, cell, n_params, seed, first and last window, pass flags, steps_to_criterion, greedy_eval, seconds_per_step, full curve, all episodes |
| `results/grid_SETTING_summary.json` | job list, exit codes, imitation pass per encoder |
| `results/compare.md` | table over cells and seeds, mean ± std, matched / not matched marks |
| `results/NAME_eval.json`, `gifs/*.gif` | evaluation metrics and GIF |
| `models/NAME.pt`, `checkpoints/jepa_NAME.pt` | actor + critic (+ JEPA parts); pretrained JEPA encoder |
| `data/transitions_SETTING.npz` | saved transitions |
| `logs/JOB.log` | output of each job |

Run names: `sa_` = single agent, `ma_` = multi agent, then the cell letter and the seed (`ma_C_s0`).
