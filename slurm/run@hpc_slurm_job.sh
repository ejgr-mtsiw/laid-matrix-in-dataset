#!/bin/bash

##CHANGE THIS!

#SBATCH --job-name="P2-1x1-bench_dataset@hpc"

#SBATCH --output=out.%x.%j
##SBATCH --error=err.%x.%j

##SBATCH --time=0:1:0

#SBATCH --ntasks=1
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1

DATASET_FILE="bench_dataset.hd5.original"
DATASET_NAME="dados"

## Dataset directories
DATASET_DIR="../datasets"
DATASET_TMP_DIR="../datasets/tmp"

## The operation changes the dataset file used so we need to make a copy
OUTPUT_DATASET_FILE="$DATASET_TMP_DIR/$(date +%s)_$DATASET_FILE"
INPUT_DATASET_FILE="$DATASET_DIR/$DATASET_FILE"

## Remove the new dataset file? YES/NO
REMOVE_DATASET="YES"

## MAYBE CHANGE THIS!

EXE="./bin/laid-hdf5-mpi"

# Disable warning for mismatched library versions
# Cirrus.8 has different hdf5 versions on short and hpc partitions
# even if we load the same module
# ##Headers are 1.14.0, library is 1.10.5
#HDF5_DISABLE_VERSION_CHECK=1 # Runs but shows warning message
HDF5_DISABLE_VERSION_CHECK=2 # Runs without showing the warning message
export HDF5_DISABLE_VERSION_CHECK

# Used to guarantee that the environment does not have any other loaded module
module purge

# Load software modules. Please check session software for the details
module load gcc11/libs/hdf5/1.14.0

# Be sure to request the correct partition to avoid the job to be held in the queue, furthermore
#	on CIRRUS-B (Minho)  choose for example HPC_4_Days
#	on CIRRUS-A (Lisbon) choose for example hpc
#SBATCH --partition=hpc

## DON'T CHANGE THIS!

# Move to base dir
cd ..

# Run
if [ -f "$INPUT_DATASET_FILE" ]; then
	echo "=== Making a copy of the dataset ==="
	echo "cp $INPUT_DATASET_FILE $OUTPUT_DATASET_FILE"
	echo
	cp $INPUT_DATASET_FILE $OUTPUT_DATASET_FILE

	if [ $? -ne 0 ]
	then
		exit 1
	fi

	echo "=== Running ==="
	if [ -f "$EXE" ]; then
		chmod u+x $EXE

		echo "mpiexec -np $SLURM_NTASKS $EXE -d $DATASET_NAME -f $OUTPUT_DATASET_FILE"
		echo
		time mpiexec -np $SLURM_NTASKS $EXE -d $DATASET_NAME -f $OUTPUT_DATASET_FILE
	else
		echo "$EXE not found!"
	fi

	if [ $REMOVE_DATASET == "YES" ]; then
		echo
		echo "=== Removing dataset ==="
		echo "rm $OUTPUT_DATASET_FILE"
		echo
		rm $OUTPUT_DATASET_FILE
	fi
else
	echo "Input dataset not found! [$INPUT_DATASET_FILE]"
fi

echo "Finished with job ID: $SLURM_JOBID"
