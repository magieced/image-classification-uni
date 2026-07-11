#!/bin/sh
#SBATCH --job-name=conrads-slurm-model-accuracy-testing
#SBATCH --partition=All
python3 -m venv ~/venv
~/venv/bin/pip install -r requirements.txt
~/venv/bin/python model.py