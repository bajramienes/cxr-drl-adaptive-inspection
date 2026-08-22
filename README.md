# Adaptive Sequential Region Selection for Chest X-ray Analysis

This repository contains the scripts, experimental CSV results, and figures
used in the study **Deep Reinforcement Learning for Adaptive Visual Attention in Chest X-ray Analysis**.

## Dataset

The experiments use the publicly available **NIH ChestX-ray14** dataset:

https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

The dataset is not redistributed in this repository. Download it from the
official source and prepare the patient-level data splits before running the
experiments.

## Repository structure

```text
charts/                         Manuscript figures
results/
  original_three_seeds/         Results for seeds 5, 25, and 125
  additional_seven_seeds/       Results for seeds 15, 35, 45, 55, 65, 75, and 85
scripts/
  prepare_dataset.py            Dataset preparation and patient-level splits
  run.py                        Original resumable experiment runner
  new_seeds_.py                 Additional seven-seed experiment runner
  create_summary_final_results.py  Result aggregation and summaries
```

## Experimental setup

The Deep Reinforcement Learning evaluation includes PPO, SAC, and DQN across
ten independent random seeds:

```text
5, 15, 25, 35, 45, 55, 65, 75, 85, 125
```

Training is conducted as one continuous 6,000-episode process with Early,
Mid, and Final evaluation checkpoints. The `results` directory contains the
corresponding training and evaluation CSV files.

## Execution order

```bash
python scripts/prepare_dataset.py
python scripts/run.py
python scripts/new_seeds_.py
python scripts/create_summary_final_results.py
```

The runners are resumable. The original experiment must be completed before
`new_seeds_.py`, because the additional runs reuse the shared feature caches.
