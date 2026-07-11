#!/bin/sh
#sbatch --job-name=conrads-slurm-model-accuracy-testing
#sbatch --partition=AMD
#sbatch --mem=6G
#sbash --gpus=1
python3 -m venv ~/venv
~/venv/bin/pip install -r requirements.txt
~/venv/bin/python model.py