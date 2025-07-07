#!/bin/bash

# Parameters
#SBATCH --account=tom.balay
#SBATCH --cpus-per-task=40
#SBATCH --error=/home/tom.balay/Baking_EEG/submitit/logs_submitit_jobs_LG/20250630_102701_TpSM49_LG/%j_0_log.err
#SBATCH --job-name=LG_TpSM49
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tom.balay/Baking_EEG/submitit/logs_submitit_jobs_LG/20250630_102701_TpSM49_LG/%j_0_log.out
#SBATCH --partition=CPU
#SBATCH --signal=USR2@90
#SBATCH --time=720
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/tom.balay/Baking_EEG/submitit/logs_submitit_jobs_LG/20250630_102701_TpSM49_LG/%j_%t_log.out --error /home/tom.balay/Baking_EEG/submitit/logs_submitit_jobs_LG/20250630_102701_TpSM49_LG/%j_%t_log.err /home/tom.balay/.venvs/py3.11_cluster/bin/python -u -m submitit.core._submit /home/tom.balay/Baking_EEG/submitit/logs_submitit_jobs_LG/20250630_102701_TpSM49_LG
