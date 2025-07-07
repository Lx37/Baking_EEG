#!/bin/bash

# Parameters
#SBATCH --account=tom.balay
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=40
#SBATCH --error=/home/tom.balay/Baking_EEG/submitit/logs_submitit_master/slurm_jobs_ALL_8_GROUPS_INTRA_20250703_132217/%A_%a_0_log.err
#SBATCH --exclusive
#SBATCH --job-name=submitit
#SBATCH --mem=75G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tom.balay/Baking_EEG/submitit/logs_submitit_master/slurm_jobs_ALL_8_GROUPS_INTRA_20250703_132217/%A_%a_0_log.out
#SBATCH --partition=CPU
#SBATCH --signal=USR2@90
#SBATCH --time=7200000
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/tom.balay/Baking_EEG/submitit/logs_submitit_master/slurm_jobs_ALL_8_GROUPS_INTRA_20250703_132217/%A_%a_%t_log.out --error /home/tom.balay/Baking_EEG/submitit/logs_submitit_master/slurm_jobs_ALL_8_GROUPS_INTRA_20250703_132217/%A_%a_%t_log.err /home/tom.balay/.venvs/py3.11_cluster/bin/python -u -m submitit.core._submit /home/tom.balay/Baking_EEG/submitit/logs_submitit_master/slurm_jobs_ALL_8_GROUPS_INTRA_20250703_132217
