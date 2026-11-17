#!/bin/bash
# Robometer on 8 episodes (2 successes and 2 failures per task), to measure seconds per episode.
# Scores the same 8 episodes twice, with 16 pictures and with 8, then compares the two
# (our mentor asked for evidence that 16 pictures are fine for Robometer).
# Submit from the repo folder on Lonestar6:   sbatch week3/tacc/robometer_test.slurm
#SBATCH -J robometer-test
#SBATCH -o robometer-test.%j.out
#SBATCH -e robometer-test.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION

export HF_HOME=$SCRATCH/hf-cache
source $WORK/robometer/.venv/bin/activate
cd $WORK/failure-detector-for-oopsiedata
nvidia-smi
python week3/tacc/robometer_batch.py --data $WORK/oopsie --limit 2 --out week3/robometer_test.jsonl
python week3/tacc/robometer_batch.py --data $WORK/oopsie --limit 2 --frames 8 --out week3/robometer_test_8.jsonl
python week3/tacc/compare_robometer_frames.py
