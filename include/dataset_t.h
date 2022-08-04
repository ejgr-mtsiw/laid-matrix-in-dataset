/*
 ============================================================================
 Name        : dataset_t.h
 Author      : Eduardo Ribeiro
 Description : Datatype representing one dataset
 ============================================================================
 */

#ifndef DATASET_T_H
#define DATASET_T_H

#include "word_t.h"

#include <stdint.h>

typedef struct dataset_t
{
	/**
	 * Number of attributes
	 */
	uint32_t n_attributes;

	/**
	 * Number of words needed to store a line
	 */
	uint32_t n_words;

	/**
	 * Number of bits needed to store jnsqs (max 32)
	 */
	uint8_t n_bits_for_jnsqs;

	/**
	 * Number of observations
	 */
	uint32_t n_observations;

	/**
	 * Number of classes
	 */
	uint32_t n_classes;

	/**
	 * Number of bits used to store the class (max 32)
	 */
	uint8_t n_bits_for_class;

	/**
	 * Dataset data
	 */
	word_t* data;

	/**
	 * Array with number of observations per class
	 */
	uint32_t* n_observations_per_class;

	/**
	 * Array with the indexes for each observation per class.
	 * They reference lines in *data
	 */
	uint32_t* observations_per_class;

} dataset_t;

#endif // DATASET_T_H
