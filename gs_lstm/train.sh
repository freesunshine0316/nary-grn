#!/bin/bash
#SBATCH -J c_m_4 --partition=gpu --gres=gpu:1 -C K20X --time=1:00:00 --output=train.out_c_m_4 --error=train.err_c_m_4
#SBATCH --mem=20GB
#SBATCH -c 5

python G2S_trainer.py --config_path config.json

