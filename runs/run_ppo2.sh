#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=epipolicy_hyperparameter_opt_ppo3
#SBATCH --output=epipolicy_hyperparameter_opt_ppo3.out

source ./venv/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

time python3 train.py --algo ppo --env EpiEnv-v0 -n 50000 -optimize --sampler skopt --pruner median --optimization-log-path summaries/EpiEnv-v0_2/ppo/ --tensorboard-log summaries --n-trials 1000 --n-jobs 2 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 --save-freq 10000 