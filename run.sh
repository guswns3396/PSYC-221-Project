#!/usr/bin/bash
module load viz
module load py-numpy/1.24.2_py39
module load py-matplotlib/3.7.1_py39
module load py-jupyter/1.0.0_py39
module load python/3.9.0

python3 --version
python3 segment.py > log_file 2>&1 &
