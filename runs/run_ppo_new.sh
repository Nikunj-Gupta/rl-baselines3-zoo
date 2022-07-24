#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=120:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=epipolicy_hyperparameter_opt_ppo2_new
#SBATCH --output=epipolicy_hyperparameter_opt_ppo2_new.out

source ./venv/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

time python3 train.py --algo ppo --env EpiEnv-v0 -n 30000 -optimize --sampler tpe --pruner median --optimization-log-path summaries/EpiEnv-v0_new/ppo/ --tensorboard-log summaries/ppo_new --n-trials 1000 --n-jobs 2 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 --save-freq 10000 