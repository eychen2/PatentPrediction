#!/usr/bin/env bash
#SBATCH --job-name=text_transform
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160gb
#SBATCH --time=96:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --output=/usr/xtmp/eyc14/project/exelogs/%A_%a.out
#SBATCH --error=/usr/xtmp/eyc14/project/exelogs/%A_%a.err
source /usr/xtmp/eyc14/project/dlhw/bin/activate

python3 /usr/xtmp/eyc14/project/train_text_transformer_parallel.py