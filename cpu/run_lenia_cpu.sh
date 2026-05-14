#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --output=1024/lenia_out.log
#SBATCH --time=02:00:00

#LOAD MODULES 
module load CUDA

#BUILD
make

#RUN
srun ./lenia.out
srun ./lenia.out
srun ./lenia.out
srun ./lenia.out
srun ./lenia.out