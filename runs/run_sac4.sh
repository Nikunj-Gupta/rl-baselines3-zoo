#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32 
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=epipolicy_hyperparameter_opt_sac5
#SBATCH --output=epipolicy_hyperparameter_opt_sac5.out

source ./venv/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

time python3 train.py --algo sac --env EpiEnv-v0 -n 50000 -optimize --sampler tpe --pruner halving --optimization-log-path summaries/EpiEnv-v0_4/sac/ --tensorboard-log summaries --n-trials 1000 --n-jobs 32 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 --save-freq 10000 