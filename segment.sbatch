#!/usr/bin/bash
#SBATCH --job-name=segment
#SBATCH --output=segment.final2.out
#SBATCH --error=segment.final2.err
#SBATCH --time=7:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32GB

module load viz
module load py-numpy/1.24.2_py39
module load py-matplotlib/3.7.1_py39
module load py-jupyter/1.0.0_py39
module load python/3.9.0

python3 segment.py
