#!/bin/bash
#SBATCH --job-name=train_ssd
#SBATCH --output=train_ssd.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu

module load anaconda

source ~/.bashrc

conda init bash

source ~/.bashrc
conda activate safe
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu


# main result
python launcher/examples/train_offline.py --env_id 0 --config configs/train_config.py:fisor


# data quantity
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:fisor --ratio 0.1
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:fisor --ratio 0.5

# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:fisor --ratio 0.1
# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:fisor --ratio 0.5
