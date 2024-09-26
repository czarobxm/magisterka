#!/bin/bash

srun --output=output/"test_$(date +%F--%T).txt" \
 --partition=common \
 --qos=1gpu1d \
 --time=1-0 \
 --gres=gpu:1 \
 poetry run python3 train_single_gpu.py \
 --task sequence_modelling \
 --dataset enwik9  \
 --criterion cross_entropy \
 --model decoder_only \
 --device cuda \
 --mha_type vanilla \
 --act_fun none \
 --structure 8x4096 \
 --batch_size 8 \
 --epochs 2 \
 --lr 0.0001 \
 --tokenizer google/byt5-small \
 --post_norm false