#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=20
#SBATCH --job-name=test_decimate_quick_20250709_110815
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tom.balay/Baking_EEG/submitit/logs_test_decimate_folds_20250709_110815/%j_0_log.out
#SBATCH --partition=cpu
#SBATCH --signal=USR2@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/tom.balay/Baking_EEG/submitit/logs_test_decimate_folds_20250709_110815/%j_%t_log.out /home/tom.balay/.venvs/py3.11_cluster/bin/python -u -m submitit.core._submit /home/tom.balay/Baking_EEG/submitit/logs_test_decimate_folds_20250709_110815
