#!/bin/bash

# Parameters
#SBATCH --account=tom.balay
#SBATCH --cpus-per-task=40
#SBATCH --error=/home/tom.balay/Baking_EEG/submitit/logs_decimate_folds_20250709_164702/%j_0_log.err
#SBATCH --job-name=decimate_folds_analysis
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tom.balay/Baking_EEG/submitit/logs_decimate_folds_20250709_164702/%j_0_log.out
#SBATCH --partition=CPU
#SBATCH --signal=USR2@90
#SBATCH --time=720
#SBATCH --wckey=submitit

# setup
module load python/3.11
source ~/.venvs/py3.11_cluster/bin/activate

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/tom.balay/Baking_EEG/submitit/logs_decimate_folds_20250709_164702/%j_%t_log.out --error /home/tom.balay/Baking_EEG/submitit/logs_decimate_folds_20250709_164702/%j_%t_log.err /home/tom.balay/.venvs/py3.11_cluster/bin/python -u -m submitit.core._submit /home/tom.balay/Baking_EEG/submitit/logs_decimate_folds_20250709_164702
