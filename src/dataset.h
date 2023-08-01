/*
 ============================================================================
 Name        : dataset.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage datasets
 ============================================================================
 */

#ifndef DATASET_H
#define DATASET_H

#include "types/dataset_t.h"
#include "types/oknok_t.h"
#include "types/word_t.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define DATASET_INVALID_DIMENSIONS		1
#define DATASET_NOT_ENOUGH_CLASSES		2
#define DATASET_NOT_ENOUGH_ATTRIBUTES	4
#define DATASET_NOT_ENOUGH_OBSERVATIONS 8
#define DATASET_ERROR_ALLOCATING_DATA	16

#define NEXT_LINE(line, n_words) ((line) += (n_words))

#define GET_NEXT_LINE(line, n_words) ((line) + (n_words))
#define GET_PREV_LINE(line, n_words) ((line) - (n_words))
#define GET_LAST_LINE(dset, n_lines, n_words)                                  \
	((dset) + ((n_lines - 1) * n_words))

/**
 * Initializes a dataset structure
 */
void init_dataset(dataset_t* dataset);

/**
 * Returns the class of this data line
 */
uint32_t get_class(const word_t* line, const uint32_t n_attributes,
				   const uint32_t n_words, const uint8_t n_bits_for_class);

/**
 * Compares two lines of the dataset
 * Used to sort the dataset
 */
// int compare_lines(const void *a, const void *b);
int compare_lines_extra(const void* a, const void* b, void* n_words);

/**
 * Checks if the lines have the same attributes
 */
bool has_same_attributes(const word_t* a, const word_t* b,
						 const uint32_t n_attributes);

/**
 * Removes duplicated lines from the dataset.
 * Assumes the dataset is ordered
 * Returns number of removed observations
 */
uint32_t remove_duplicates(dataset_t* dataset);

/**
 * Fill the arrays with the number of items per class and also a matrix with
 * references to the lines that belong to each class to simplify the
 * calculation of the disjoint matrix
 */
oknok_t fill_class_arrays(dataset_t* dataset);

/**
 * Frees dataset memory
 */
void free_dataset(dataset_t* dataset);

#endif
