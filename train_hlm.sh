#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
mkdir -p ../test_results/noisy_hlm
python main.py --data ../input/hlm/train/ --save ../test_results/noisy_hlm/ --train_percent 0.85 --epoch 5 --second-gcn-dimensions 16 --batch_size 32 --mi_weight 0.001 --con_weight 5 --noise_scale 0.01