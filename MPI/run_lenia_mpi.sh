#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --output=1024/32/lenia_out.log
#SBATCH --hint=nomultithread

#Load MPI module 
module load OpenMPI

make clean
make


mpirun -np $SLURM_NTASKS --mca pml ob1 --mca btl tcp,self ./lenia.out
echo "RUN 1 DONE"

mpirun -np $SLURM_NTASKS --mca pml ob1 --mca btl tcp,self ./lenia.out
echo "RUN 2 DONE"

mpirun -np $SLURM_NTASKS --mca pml ob1 --mca btl tcp,self ./lenia.out
echo "RUN 3 DONE"

mpirun -np $SLURM_NTASKS --mca pml ob1 --mca btl tcp,self ./lenia.out
echo "RUN 4 DONE"

mpirun -np $SLURM_NTASKS --mca pml ob1 --mca btl tcp,self ./lenia.out
echo "RUN 5 DONE"
