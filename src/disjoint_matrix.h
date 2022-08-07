/*
 ============================================================================
 Name        : disjoint_matrix.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix
 ============================================================================
 */

#ifndef DISJOINT_MATRIX_H
#define DISJOINT_MATRIX_H

#include "types/dataset_t.h"
#include "types/oknok_t.h"

#include "hdf5.h"

#include <stdbool.h>
#include <stdint.h>

/**
 * Calculates the number of lines for the disjoint matrix
 */
uint32_t get_dm_n_lines(const dataset_t* dataset);

/**
 * Writes the matrix atributes in the dataset
 */
herr_t write_dm_attributes(const hid_t dataset_id, const uint32_t n_attributes,
						   const uint32_t n_matrix_lines);

#endif
