#!/bin/bash

#SBATCH --job-name=lr_search
#SBATCH --time=1-0
#SBATCH --partition=common
#SBATCH --qos=1gpu1d
#SBATCH --gres=gpu:1
#SBATCH --output=output/lr_search_%j.out


param1=$1 # mha type
param2=$2 # lr
param3=$3 # structure

cosformer="cosformer"
if [ "$param1" == "$cosformer" ]; then
    act_fun="relu"
else
    act_fun="none"
fi

poetry run python3 train_single_gpu.py \
 --task sequence_modelling \
 --dataset enwik9  \
 --criterion cross_entropy \
 --model decoder_only \
 --device cuda \
 --mha_type $param1 \
 --act_fun $act_fun \
 --structure $param3 \
 --batch_size 2 \
 --epochs 2 \
 --init_lr $param2 \
 --tokenizer google/byt5-small \
 --post_norm false


