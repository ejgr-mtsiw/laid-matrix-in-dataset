#!/bin/bash

#SBATCH --job-name="P2-compile@short"

##SBATCH --time=0:1:0

#SBATCH --ntasks=1
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1

# Be sure to request the correct partition to avoid the job to be held in the queue, furthermore
#	on CIRRUS-B (Minho)  choose for example HPC_4_Days
#	on CIRRUS-A (Lisbon) choose for example hpc
#SBATCH --partition=short

#SBATCH --output=out.%x.%j
#SBATCH --error=err.%x.%j

# Used to guarantee that the environment does not have any other loaded module
module purge

# Load software modules. Please check session software for the details
module load gcc11/libs/hdf5/1.14.0

# Compile application
cd ..
make clean
make release

echo "Finished with job $SLURM_JOBID"
