#!/bin/bash

# Parameters
#SBATCH --account=tom.balay
#SBATCH --cpus-per-task=40
#SBATCH --error=/home/tom.balay/Baking_EEG/submitit/submitit_logs/TpSM49_PP/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tom.balay/Baking_EEG/submitit/submitit_logs/TpSM49_PP/%j_0_log.out
#SBATCH --partition=CPU
#SBATCH --signal=USR2@90
#SBATCH --time=720000
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/tom.balay/Baking_EEG/submitit/submitit_logs/TpSM49_PP/%j_%t_log.out --error /home/tom.balay/Baking_EEG/submitit/submitit_logs/TpSM49_PP/%j_%t_log.err /home/tom.balay/.venvs/py3.11_cluster/bin/python -u -m submitit.core._submit /home/tom.balay/Baking_EEG/submitit/submitit_logs/TpSM49_PP
