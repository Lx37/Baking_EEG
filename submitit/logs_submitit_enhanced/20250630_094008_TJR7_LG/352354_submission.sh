#!/bin/bash

# Parameters
#SBATCH --account=tom.balay
#SBATCH --cpus-per-task=40
#SBATCH --error=logs_submitit_enhanced/20250630_094008_TJR7_LG/slurm-%j.err
#SBATCH --job-name=LG_TJR7
#SBATCH --job-name=submitit
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=logs_submitit_enhanced/20250630_094008_TJR7_LG/slurm-%j.out
#SBATCH --partition=CPU
#SBATCH --signal=USR2@90
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/tom.balay/Baking_EEG/submitit/logs_submitit_enhanced/20250630_094008_TJR7_LG/%j_%t_log.out --error /home/tom.balay/Baking_EEG/submitit/logs_submitit_enhanced/20250630_094008_TJR7_LG/%j_%t_log.err /home/tom.balay/.venvs/py3.11_cluster/bin/python -u -m submitit.core._submit /home/tom.balay/Baking_EEG/submitit/logs_submitit_enhanced/20250630_094008_TJR7_LG
