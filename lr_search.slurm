#!/bin/bash

#SBATCH --job-name=lr_search
#SBATCH --time=1-0
#SBATCH --partition=common
#SBATCH --qos=4gpu1d
#SBATCH --gres=gpu:1
#SBATCH --output=output/lr_search_%j.out


mha_type=$1 # mha type
lr=$2 # lr
structure=$3 # structure
max_length=$4 # max_length
batch_size=$5 # batch_size
gradient_accumulation_steps=$6 # gradient_accumulation_steps
hourglass_downsampling_type=$7 # hourglass_downsampling_type
hourglass_upsampling_type=$8 # hourglass_upsampling_type
hourglass_attention_downsampling=$9 # hourglass_attention_downsampling
hourglass_attention_upsampling=${10} # hourglass_attention_upsampling
hourglass_upsampling_residual=${11} # hourglass_upsampling_residual
hourglass_sampling_post_norm=${12} # hourglass_sampling_post_norm
hourglass_sampling_use_linear=${13} # hourglass_sampling_use_linear
hourglass_sampling_use_feedforward=${14} # hourglass_sampling_use_feedforward


cosformer="cosformer"
if [ "$mha_type" == "$cosformer" ]; then
    act_fun="relu"
else
    act_fun="none"
fi

echo $structure
echo $batch_size
echo $hourglass_downsampling_type
echo $hourglass_upsampling_type

poetry run python3 train_single_gpu.py \
 --task sequence_modelling \
 --dataset enwik9  \
 --criterion cross_entropy \
 --model decoder_only \
 --device mps \
 --mha_type $mha_type \
 --act_fun $act_fun \
 --structure $structure \
 --max_length $max_length \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --epochs 1 \
 --init_lr $lr \
 --tokenizer google/byt5-small \
 --hourglass_downsampling_type $hourglass_downsampling_type \
 --hourglass_upsampling_type $hourglass_upsampling_type \
 --hourglass_attention_downsampling $hourglass_attention_downsampling \
 --hourglass_attention_upsampling $hourglass_attention_upsampling \
 --hourglass_upsampling_residual $hourglass_upsampling_residual \
 --hourglass_sampling_post_norm $hourglass_sampling_post_norm \
 --hourglass_sampling_use_linear $hourglass_sampling_use_linear \
 --hourglass_sampling_use_feedforward $hourglass_sampling_use_feedforward