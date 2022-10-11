#!/bin/bash

##CHANGE THIS!

#SBATCH --job-name="laid-mpi-bench_dataset@hpc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=0:1:0

## The operation changes the dataset file used so we need to make a copy
INPUT_DATASET_FILE="../datasets/bench_dataset.hd5.original"
OUTPUT_DATASET_FILE="../bench_dataset.h5"
DATASET_NAME="dados"

## Remove the new dataset file? YES/NO
REMOVE_DATASET="YES"

## MAYBE CHANGE THIS!

EXE="../bin/laid-hdf5-mpi"

# Used to guarantee that the environment does not have any other loaded module
module purge

# Load software modules. Please check session software for the details
module load hdf5/1.12.0
##module load gcc-8.3
##module load clang/7.0.0

## DON'T CHANGE THIS!

# Be sure to request the correct partition to avoid the job to be held in the queue, furthermore
#	on CIRRUS-B (Minho)  choose for example HPC_4_Days
#	on CIRRUS-A (Lisbon) choose for example hpc
#SBATCH --partition=hpc

# Run
echo "=== Copy dataset ==="
if [ -f "$INPUT_DATASET_FILE" ]; then
    cp $INPUT_DATASET_FILE $OUTPUT_DATASET_FILE
    
    echo "=== Running ==="
    if [ -f "$EXE" ]; then
        chmod u+x $EXE
        $EXE -d $DATASET_NAME -f $OUTPUT_DATASET_FILE
    fi

    if [ $REMOVE_DATASET == "YES" ]; then
        echo "=== Removing dataset ==="
        rm $OUTPUT_DATASET_FILE
    fi
else
    echo "Input dataset not found! [$INPUT_DATASET_FILE]"
fi

echo "Finished with job $SLURM_JOBID"
