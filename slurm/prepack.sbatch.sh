#!/bin/bash
# Offline tokenization + packing (src/data/prepack.py), CPU only.
#   ./run.sh slurm/prepack.sbatch.sh <config.yaml>
#SBATCH -A FAIR_NLP
#SBATCH -p lrd_all_serial
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python src/data/prepack.py "$1"
