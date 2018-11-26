#!/bin/bash
#SBATCH -C K20X --partition=gpu --gres=gpu:1 --time=1:00:00 --output=decode.out_c_b_0 --error=decode.err_c_b_0
#SBATCH --mem=10GB
#SBATCH -c 6

start=`date +%s`
python G2S_evaluater.py --model_prefix logs/G2S.cross_bin_0 \
        --in_path data/dev_list_0 \
        --out_path logs/results_c_b_0.json

end=`date +%s`
runtime=$((end-start))
echo $runtime
