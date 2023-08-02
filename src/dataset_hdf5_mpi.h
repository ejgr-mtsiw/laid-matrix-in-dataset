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
#include "types/word_t.h"

#include "hdf5.h"
#include "mpi.h"

#include <stdint.h>

/**
 * Opens the file and dataset indicated
 */
oknok_t mpi_hdf5_open_dataset(const char* filename, const char* datasetname,
							  const MPI_Comm comm, const MPI_Info info,
							  dataset_hdf5_t* dataset);

/**
 * Creates a new dataset in the indicated file
 */
hid_t create_hdf5_dataset(const hid_t file_id, const char* name,
						  const uint32_t n_lines, const uint32_t n_words,
						  const hid_t datatype);

/**
 * Writes n_lines_out to the dataset
 */
oknok_t write_n_lines(const hid_t dset_id, const uint32_t start,
					  const uint8_t n_lines_out, const uint32_t n_words,
					  const hid_t datatype, const void* buffer);

#endif // MPI_HDF5_DATASET_H
