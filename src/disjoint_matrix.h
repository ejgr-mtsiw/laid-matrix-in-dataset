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
 * Checks if the disjoint matrix datasets are already present in the hdf5 file
 */
bool is_matrix_created(const char* filename);

/**
 * Builds the disjoint matrix and saves it to the hdf5 dataset file
 * It will build and store 2 datasets one with attributes as lines
 * the other with atributes as columns
 */
oknok_t create_disjoint_matrix(const char* filename, const dataset_t* dataset);

/**
 * Creates the dataset containing the disjoint matrix with attributes as columns
 */
oknok_t create_line_dataset(const hid_t file_id, const dataset_t* dataset);

/**
 * Creates the dataset containing the disjoint matrix with attributes as lines
 */
oknok_t create_column_dataset(const hid_t file_id, const dataset_t* dataset);

/**
 * Writes the matrix atributes in the dataset
 */
herr_t write_dm_attributes(const hid_t dataset_id, const uint32_t n_attributes,
						   const uint32_t n_matrix_lines);

/**
 * Writes the line totals metadata to the dataset
 */
herr_t write_line_totals(const hid_t file_id, const uint32_t* data,
						 const uint32_t n_lines);

/**
 * Writes the attribute totals metadata to the dataset
 */
herr_t write_attribute_totals(const hid_t file_id, const uint32_t* data,
							  const uint32_t n_attributes);

#endif
