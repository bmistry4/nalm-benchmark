#!/bin/sh

export CUDA_VERSION='10.1'
export CUDNN_VERSION='7.6.0'
export TENSORBOARD_DIR=../../../../../datasets/bm4g15/nalu-stable-exp/tensorboard
export SAVE_DIR=../../../../../datasets/bm4g15/nalu-stable-exp/saves

#module load python3
#module load gcc/4.9.2
#module load cuda/$CUDA_VERSION
#module load cudnn/v$CUDNN_VERSION-prod-cuda-$CUDA_VERSION

#export PYTHONPATH=./ 
#conda activate ~/miniconda3/envs/nalu-env$

python3 -u "$@"