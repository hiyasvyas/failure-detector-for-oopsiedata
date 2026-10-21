#!/bin/bash
# Robometer on 8 episodes (2 successes and 2 failures per task), to measure seconds per episode.
# Submit from the repo folder on Lonestar6:   sbatch week3/tacc/robometer_test.slurm
#SBATCH -J robometer-test
#SBATCH -o robometer-test.%j.out
#SBATCH -e robometer-test.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A YOUR_ALLOCATION

export HF_HOME=$SCRATCH/hf-cache
source $WORK/robometer/.venv/bin/activate
cd $WORK/failure-detector-for-oopsiedata
nvidia-smi
python week3/tacc/robometer_batch.py --data $WORK/oopsie --limit 2 --out week3/robometer_test.jsonl
