/*
 ============================================================================
 Name        : disjoint_matrix_mpi.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix in MPIIO
 ============================================================================
 */

#ifndef MPI_DISJOINT_MATRIX_H
#define MPI_DISJOINT_MATRIX_H

#include "types/dataset_hdf5_t.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/oknok_t.h"

#include <stdint.h>

/**
 * Number of lines to buffer before output
 * Only used in the line dataset, because we already write WORD_BITS columns at
 * a time in the column dataset
 */
#define N_LINES_OUT 42

/**
 * Generates the steps for the partial disjoint matrix dm
 */
oknok_t generate_steps(const dataset_t* dataset, dm_t* dm);

/**
 * Creates the dataset containing the disjoint matrix with attributes as columns
 */
oknok_t mpi_create_line_dataset(const dataset_hdf5_t* hdf5_dset,
								const dataset_t* dset, const dm_t* dm);

/**
 * Creates the dataset containing the disjoint matrix with attributes as
 * lines
 */
oknok_t mpi_create_column_dataset(const dataset_hdf5_t* hdf5_dset,
								  const dataset_t* dset, const int rank,
								  const int size);

/**
 * Writes the attribute totals metadata to the dataset
 */
oknok_t mpi_write_attribute_totals(const hid_t dataset_id, const uint32_t start,
								   const uint32_t n_attributes,
								   const uint32_t* data);

#endif // MPI_DISJOINT_MATRIX_H
