#!/bin/bash

#SBATCH --job-name=lr_search
#SBATCH --time=1-0
#SBATCH --partition=common
#SBATCH --qos=4gpu1d
#SBATCH --gres=gpu:1
#SBATCH --output=output/lr_search_%j.out


param1=$1 # mha type
param2=$2 # lr
param3=$3 # structure
param4=$4 # max_length
param5=$5 # batch_size
param6=$6 # gradient_accumulation_steps
param7=$7 # hourglass_downsampling_type
param8=$8 # hourglass_upsampling_type
param9=$9 # hourglass_attention_downsampling
param10=${10} # hourglass_attention_upsampling
param11=${11} # hourglass_upsampling_residual


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
 --max_length $param4 \
 --batch_size $param5 \
 --gradient_accumulation_steps $param6 \
 --epochs 1 \
 --init_lr $param2 \
 --tokenizer google/byt5-small \
 --hourglass_downsampling_type $param7 \
 --hourglass_upsampling_type $param8 \
 --hourglass_attention_downsampling $param9 \
 --hourglass_attention_upsampling $param10 \
 --hourglass_upsampling_residual $param11 \


