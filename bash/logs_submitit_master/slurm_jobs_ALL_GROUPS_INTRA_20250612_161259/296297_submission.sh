#!/bin/bash

# Parameters
#SBATCH --account=tom.balay
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=40
#SBATCH --error=/home/tom.balay/Baking_EEG/bash/logs_submitit_master/slurm_jobs_ALL_GROUPS_INTRA_20250612_161259/%A_%a_0_log.err
#SBATCH --job-name=submitit
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tom.balay/Baking_EEG/bash/logs_submitit_master/slurm_jobs_ALL_GROUPS_INTRA_20250612_161259/%A_%a_0_log.out
#SBATCH --partition=CPU
#SBATCH --signal=USR2@90
#SBATCH --time=3600
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/tom.balay/Baking_EEG/bash/logs_submitit_master/slurm_jobs_ALL_GROUPS_INTRA_20250612_161259/%A_%a_%t_log.out --error /home/tom.balay/Baking_EEG/bash/logs_submitit_master/slurm_jobs_ALL_GROUPS_INTRA_20250612_161259/%A_%a_%t_log.err /home/tom.balay/.venvs/py3.11_cluster/bin/python -u -m submitit.core._submit /home/tom.balay/Baking_EEG/bash/logs_submitit_master/slurm_jobs_ALL_GROUPS_INTRA_20250612_161259
