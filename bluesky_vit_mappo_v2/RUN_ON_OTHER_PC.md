# Run bluesky_vit_mappo_v2 on another PC

Base commit of the repo: see below. Folder layout needed:
  <repo>/               (AI4REAL-NET-Competition branch, with bluesky_gym, bluesky_zoo, core)
  <repo>/bluesky_vit_mappo_v2/   (this folder)

## Setup (once)
    git clone -b AI4REAL-NET-Competition https://github.com/TUDelft-CNS-ATM/bluesky-gym.git bluesky-gym
    cd bluesky-gym && uv sync && uv add torch      # skip "uv add torch" if torch is in the lock file
    tar xzf bluesky_vit_mappo_v2.tar.gz            # creates bluesky_vit_mappo_v2/
    .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
Use --device cuda only if the GPU shows up in that arch list. Otherwise use --device cpu.

## First run only: warm the BlueSky cache (avoids a race between parallel runs)
    cd bluesky-gym/bluesky_vit_mappo_v2
    PYTHONPATH=.. ../.venv/bin/python -m scripts.pretrain_encoder --transitions 100 --epochs 1 --batch-size 16 --n-agents 10 --device cpu --output checkpoints/smoke.pt

## Runs (one BlueSky env per process; run seeds as separate processes)
    # MAPPO from scratch, one seed
    PYTHONPATH=.. ../.venv/bin/python -m scripts.train_mappo --n-agents 10 --steps 100000 --seed 42 --device cpu --output models/mappo_scratch_seed42.pt
    # optional pretraining (use the same --n-agents as MAPPO)
    PYTHONPATH=.. ../.venv/bin/python -m scripts.pretrain_encoder --transitions 20000 --epochs 5 --batch-size 128 --n-agents 10 --device cpu --output checkpoints/enc_10ag.pt
    # MAPPO from the pretrained encoder
    PYTHONPATH=.. ../.venv/bin/python -m scripts.train_mappo --n-agents 10 --steps 100000 --seed 42 --device cpu --pretrained checkpoints/enc_10ag.pt --output models/mappo_pre_seed42.pt
    # evaluation
    PYTHONPATH=.. ../.venv/bin/python -m scripts.evaluate_mappo --model models/mappo_scratch_seed42.pt --episodes 20 --n-agents 10 --device cpu

## Notes
- Token count depends on --n-agents (3 aircraft: 21 tokens, 10 aircraft: 28). Pretrain and MAPPO must use the same value.
- Measured on the source PC (i7-8750H, CPU): MAPPO about 0.09 s per step; pretraining about 116 s per epoch for 2000 transitions.
- Official scoring: scripts.evaluate_competition --env ma (1000 episodes, seed 42). evaluate_mappo is a quick check only.
