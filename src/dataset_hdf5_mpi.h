/*
 ============================================================================
 Name        : dataset_hdf5_mpi.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage hdf5 datasets using MPIIO
 ============================================================================
 */

#ifndef MPI_HDF5_DATASET_H
#define MPI_HDF5_DATASET_H

#include "types/dataset_hdf5_t.h"
#include "types/oknok_t.h"

#include "hdf5.h"
#include "mpi.h"

#include <stdint.h>

/**
 * Opens the file and dataset indicated
 */
oknok_t mpi_hdf5_open_dataset(const char* filename, const char* datasetname,
							  const MPI_Comm comm, const MPI_Info info,
							  dataset_hdf5_t* dataset);

#endif // MPI_HDF5_DATASET_H
