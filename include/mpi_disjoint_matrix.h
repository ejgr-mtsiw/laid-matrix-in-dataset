/*
 ============================================================================
 Name        : mpi_disjoint_matrix.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix in MPIIO
 ============================================================================
 */

#ifndef MPI_DISJOINT_MATRIX_H
#define MPI_DISJOINT_MATRIX_H

#include "disjoint_matrix.h"
#include "hdf5_dataset.h"

#include <stdint.h>

uint32_t roundUp(uint32_t numToRound, uint32_t multiple);

/**
 * Builds the disjoint matrix and saves it to the hdf5 dataset file
 * It will build and store 2 datasets one with attributes as lines
 * the other with atributes as columns
 */
// oknok_t mpi_create_disjoint_matrix(const char* filename,
//								   const dataset_t* dataset, int mpi_rank,
//								   int mpi_size, MPI_Comm comm, MPI_Info info);

/**
 * Creates the dataset containing the disjoint matrix with attributes as columns
 */
// oknok_t mpi_create_line_dataset(const hid_t file_id, const dataset_t*
// dataset, 								const int mpi_rank, const int
// mpi_size, MPI_Comm comm);

oknok_t mpi_create_line_dataset(hdf5_dataset_t hdf5_dset, word_t* data,
								const uint32_t n_matrix_lines,
								const uint32_t n_words, steps_t* steps,
								const int rank, const int size);

/**
 * Creates the dataset containing the disjoint matrix with attributes as
 * lines
 */
// oknok_t mpi_create_column_dataset(const hid_t file_id, const dataset_t*
// dataset, 								  const int mpi_rank, const int
// mpi_size);

/**
 * Writes the matrix atributes in the dataset
 */
// herr_t mpi_write_disjoint_matrix_attributes(const hid_t dataset_id,
//											const uint32_t n_attributes,
//											const uint32_t n_matrix_lines);

/**
 * Writes the line totals metadata to the dataset
 */
// herr_t mpi_write_line_totals(const hid_t file_id, const uint32_t* data,
//							 const uint32_t n_lines);

/**
 * Writes the attribute totals metadata to the dataset
 */
// herr_t mpi_write_attribute_totals(const hid_t file_id, const uint32_t* data,
//								  const uint32_t n_attributes);

#endif // MPI_DISJOINT_MATRIX_H
