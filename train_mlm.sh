#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
mkdir -p ../test_results/noisy_mlm
python main.py --data ../input/mlm/train/ --save ../test_results/noisy_mlm/ --train_percent 0.85 --epoch 3 --second-gcn-dimensions 16 --batch_size 32 --mi_weight 0.001 --con_weight 5 --noise_scale 0.01