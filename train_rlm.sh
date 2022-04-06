#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
mkdir -p ../test_results/noisy_rlm
python main.py --data ../input/rlm/train/ --save ../test_results/noisy_rlm/ --train_percent 0.85 --epoch 3 --second-gcn-dimensions 16 --batch_size 32 --mi_weight 0.001 --con_weight 1 --noise_scale 0.01 --property rlm