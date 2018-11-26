#!/bin/bash
#SBATCH -J bi_c_m_3 -C K20X --partition=gpu --gres=gpu:1 --time=10:00:00 --output=train.out_c_m_3 --error=train.err_c_m_3
#SBATCH --mem=20GB
#SBATCH -c 5

python G2S_trainer.py --config_path config.json

