#!/bin/bash
# Robometer on all 400 episodes. Run the test job first and set -t from its timing.
# If the job runs out of time, submit it again: finished episodes are skipped.
# If compare_robometer_frames.py says 16 pictures are worse, add --frames 8 to the python line
# (it then writes week3/robometer_scores_8.jsonl, which run_heads.py --frames 8 reads).
# Submit from the repo folder on Lonestar6:   sbatch week3/tacc/robometer_all.slurm
#SBATCH -J robometer-all
#SBATCH -o robometer-all.%j.out
#SBATCH -e robometer-all.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A YOUR_ALLOCATION

export HF_HOME=$SCRATCH/hf-cache
source $WORK/robometer/.venv/bin/activate
cd $WORK/failure-detector-for-oopsiedata
python week3/tacc/robometer_batch.py --data $WORK/oopsie
