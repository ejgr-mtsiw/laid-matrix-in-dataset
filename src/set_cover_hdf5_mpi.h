/*
 ============================================================================
 Name        : set_cover_hdf5_mpi.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to apply the set cover algorithm
			   using hdf5 and mpi
 ============================================================================
 */

#ifndef SET_COVER_HDF5_MPI_H
#define SET_COVER_HDF5_MPI_H

#include "types/oknok_t.h"
#include "types/word_t.h"

#include "hdf5.h"

#include <stdint.h>

/**
 * Reads attribute data
 */
oknok_t get_column(hid_t dataset, uint32_t attribute, uint32_t offset,
				   uint32_t count, word_t* column);

/**
 * Updates covered status of the matrix lines
 */
oknok_t update_covered_lines_mpi(const word_t* column, const uint32_t n_words,
								 word_t* covered_lines);

/**
 *
 */
oknok_t update_attribute_totals_mpi(cover_t* cover,
									dataset_hdf5_t* line_dataset,
									word_t* column);

#endif // SET_COVER_HDF5_MPI_H
