#!/bin/sh
#sbatch --job-name=conrads-slurm-test
#sbatch --partition=AMD
#sbatch --mem=6G
#sbash --gpus=1
python3 -m venv ~/venv
python slurmtest.py