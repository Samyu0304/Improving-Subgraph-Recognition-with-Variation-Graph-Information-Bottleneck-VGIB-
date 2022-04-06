#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
mkdir -p ../test_results/new_noisy_qed
python main.py --data ../input/qed/positive/train/ --save ../test_results/new_noisy_qed/ --train_percent 0.85 --epoch 2 --second-gcn-dimensions 16 --batch_size 32 --mi_weight 0.001 --con_weight 1 --noise_scale 0.01 --property qed