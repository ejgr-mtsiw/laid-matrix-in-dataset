#!/bin/bash

#SBATCH --job-name=compile@short
#SBATCH --time=0:1:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Be sure to request the correct partition to avoid the job to be held in the queue, furthermore
#	on CIRRUS-B (Minho)  choose for example HPC_4_Days
#	on CIRRUS-A (Lisbon) choose for example hpc
#SBATCH --partition=short

# Used to guarantee that the environment does not have any other loaded module
module purge

# Load software modules. Please check session software for the details
module load hdf5/1.12.0
##module load gcc83/openmpi/4.1.1
##module load clang/openmpi/4.0.3

# Compile application
cd ..
make clean
make release

echo "Finished with job $SLURM_JOBID"
