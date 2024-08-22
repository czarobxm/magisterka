#!/bin/bash

# IMDB
python train_single_gpu.py --dataset imdb --criterion cross_entropy --model classifier   --device mps --mha_type vanilla   --act_fun none --structure 6x512
python train_single_gpu.py --dataset imdb --criterion cross_entropy --model classifier   --device mps --mha_type cosformer --act_fun relu --structure 6x512


python train_single_gpu.py --task sequence_modelling --dataset wikipedia --criterion cross_entropy --model decoder_only --device mps --mha_type vanilla --structure 6x512 --batch_size 16

python train_single_gpu.py --task sequence_modelling --dataset wikipedia --criterion cross_entropy --model hourglass_decoder_only --device mps --mha_type vanilla --structure 2x512,2x256,2x512 --batch_size 16


python train_single_gpu.py --task sequence_modelling --dataset enwik8   --criterion cross_entropy --model decoder_only --device mps --mha_type cosformer --act_fun relu --structure 8x512 --batch_size 16 --epochs 8 --lr 0.00008 --tokenizer google/byt5-small
python train_single_gpu.py --task sequence_modelling --dataset enwik8    --criterion cross_entropy --model decoder_only --device mps --mha_type vanilla                  --structure 8x512 --batch_size 16 --epochs 3 --lr 0.00008  --tokenizer google/byt5-small

python train_single_gpu.py --task sequence_modelling --dataset wikipedia --criterion cross_entropy --model decoder_only --device mps --mha_type cosformer --act_fun relu --structure 8x512 --batch_size 16 --epochs 8 --lr 0.00008 --tokenizer bert-base-uncased
python train_single_gpu.py --task sequence_modelling --dataset wikipedia --criterion cross_entropy --model decoder_only --device mps --mha_type vanilla                  --structure 8x512 --batch_size 16 --epochs 3 --lr 0.00008  --tokenizer bert-base-uncased
