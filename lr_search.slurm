#!/bin/bash

#SBATCH --job-name=lr_search
#SBATCH --time=1-0
#SBATCH --partition=common
#SBATCH --qos=1gpu1d
#SBATCH --gres=gpu:1
#SBATCH --output=output/"test_$(date +%F--%T).txt"

param1=$1
cosformer="cosformer"

if [ "$param1" == "$cosformer" ]; then
    act_fun="relu"
else
    act_fun="none"
fi
param2=$2

poetry run python3 train_single_gpu.py \
 --task sequence_modelling \
 --dataset enwik9  \
 --criterion cross_entropy \
 --model decoder_only \
 --device cuda \
 --mha_type $param1 \
 --act_fun $act_fun \
 --structure 8x4096 \
 --batch_size 8 \
 --epochs 2 \
 --lr $param2 \
 --tokenizer google/byt5-small \
 --post_norm false
