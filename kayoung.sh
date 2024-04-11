#!/bin/bash

#SBATCH --j=mvi_vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=30G
#SBATCH -w augi2
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o logs/%N_%x_%j.out
#SBTACH -e %x_%j.err

pwd
which python
hostname

source /data/kayoung/init.sh
conda activate hi-vae

python /local_datasets/hi-vae/mvi_main.py