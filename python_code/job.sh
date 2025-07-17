#!/bin/bash
#SBATCH --job-name=job
#SBATCH --ntasks=1 --nodes=1 -- cpus-per-task=36
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL

module purge
module load miniconda

conda activate py3_base
python code/master.py