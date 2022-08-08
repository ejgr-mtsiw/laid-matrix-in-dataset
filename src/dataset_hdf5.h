/*
 ============================================================================
 Name        : dataset_hdf5.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage hdf5 datasets
 ============================================================================
 */

#ifndef HDF5_DATASET_H
#define HDF5_DATASET_H

#include "types/dataset_hdf5_t.h"
#include "types/dataset_t.h"
#include "types/oknok_t.h"

#include "hdf5.h"

#include <stdint.h>

/**
 * The name of the dataset that will store the disjoint matrix
 * with attributes as lines
 */
#define DM_COLUMN_DATA "/COLUMN_DATA"

/**
 * The name of the dataset that will store the disjoint matrix
 * with attributes as columns
 */
#define DM_LINE_DATA "/LINE_DATA"

/**
 * The name of the dataset that will store the totals for each line
 */
#define DM_LINE_TOTALS "/LINE_TOTALS"

/**
 * The name of the dataset that will dtore the attribute totals
 */
#define DM_ATTRIBUTE_TOTALS "/ATTRIBUTE_TOTALS"

/**
 * Attribute for number of classes
 */
#define N_CLASSES_ATTR "n_classes"

/**
 * Attribute for number of attributes
 */
#define N_ATTRIBUTES_ATTR "n_attributes"

/**
 * Attribute for number of observations
 */
#define N_OBSERVATIONS_ATTR "n_observations"

/**
 * Attrinute for the number of lines of the disjoint matrix
 */
#define N_MATRIX_LINES_ATTR "n_matrix_lines"

/**
 * Checks if dataset is present in file_id
 */
bool hdf5_dataset_exists(const hid_t file_id, const char* dataset);

/**
 * Checks if dataset is present from filename
 */
bool hdf5_file_has_dataset(const char* filename, const char* datasetname);

/**
 * Reads the dataset attributes from the hdf5 file
 */
oknok_t hdf5_read_dataset_attributes(hid_t dataset_id, dataset_t* dataset);

/**
 * Reads the value of one attribute from the dataset
 */
oknok_t hdf5_read_attribute(hid_t dataset_id, const char* attribute,
							hid_t datatype, void* value);

/**
 * Reads the entire dataset data from the hdf5 file
 */
oknok_t hdf5_read_dataset_data(hid_t dataset_id, word_t* data);

/**
 * Retrieves a line from the dataset
 */
oknok_t hdf5_read_line(const dataset_hdf5_t* dataset, const uint32_t index,
					   const uint32_t n_words, word_t* line);

/**
 * Reads n lines from the dataset
 */
oknok_t hdf5_read_lines(const dataset_hdf5_t* dataset, const uint32_t index,
						const uint32_t n_words, const uint32_t n_lines,
						word_t* lines);
/**
 * Writes an attribute to the dataset
 */
oknok_t hdf5_write_attribute(hid_t dataset_id, const char* attribute,
							 hid_t datatype, const void* value);

/**
 * Returns the dataset dimensions stored in the hdf5 dataset
 */
void hdf5_get_dataset_dimensions(hid_t dataset_id, hsize_t* dataset_dimensions);

/**
 * Free resources and closes open connections
 */
void hdf5_close_dataset(dataset_hdf5_t* dataset);

#endif
