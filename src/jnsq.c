/*
 ============================================================================
 Name        : jnsq.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage JNSQ
 ============================================================================
 */

#include "jnsq.h"

#include "dataset.h"
#include "types/dataset_t.h"
#include "types/word_t.h"
#include "utils/bit.h"

#include <math.h>
#include <stdint.h>

void set_jnsq_bits(word_t* line, uint32_t inconsistency,
				   const uint32_t n_attributes, const uint32_t n_words,
				   const uint8_t n_bits_for_class)
{
	// How many attributes remain on last word with attributes
	uint8_t remaining = n_attributes % WORD_BITS;

	// Number of jnsq bits on current word
	uint8_t n_bits = n_bits_for_class;

	if (remaining + n_bits_for_class > WORD_BITS)
	{
		// We have jnsq bits split between words

		// Number of jnsq bits on penultimate word
		n_bits = WORD_BITS - remaining;

		/**
		 * Invert inconsistency bits, because the algorithm
		 * expects the least significant bit to be on the right
		 */
		inconsistency = invert_n_bits((word_t) inconsistency, n_bits);

		line[n_words - 2]
			= set_bits(line[n_words - 2], inconsistency, 0, n_bits);

		// There's no more attributes on last word
		remaining = 0;

		// Remove used bits from inconsistency
		inconsistency >>= n_bits;

		// Number of jnsq bits that need to go on the last word
		n_bits = n_bits_for_class - n_bits;
	}

	// All remaining jnsq bits are in the same word
	uint8_t jnsq_start = WORD_BITS - remaining - n_bits;

	/**
	 * Invert inconsistency bits, because the algorithm
	 * expects the least significant bit to be on the right
	 */
	inconsistency = invert_n_bits((word_t) inconsistency, n_bits);

	line[n_words - 1]
		= set_bits(line[n_words - 1], inconsistency, jnsq_start, n_bits);
}

uint32_t add_jnsqs(dataset_t* dataset)
{
	// Current line
	word_t* current = dataset->data;

	// Previous line
	word_t* prev = current;

	// Number of attributes
	uint32_t n_attributes = dataset->n_attributes;

	// Number of longs in a line
	uint32_t n_words = dataset->n_words;

	// Number of observations in the dataset
	uint32_t n_observations = dataset->n_observations;

	// Number of bits needed to store class
	uint8_t n_bits_for_class = dataset->n_bits_for_class;

	// Last line
	word_t* last = GET_LAST_LINE(dataset->data, n_observations, n_words);

	// Inconsistency
	uint32_t inconsistency = 0;

	// Max inconsistency found
	uint32_t max_inconsistency = 0;

	// First line has jnsq=0
	set_jnsq_bits(current, 0, n_attributes, n_words, n_bits_for_class);

	// Now do the others
	for (prev = dataset->data; prev < last; NEXT_LINE(prev, n_words))
	{
		NEXT_LINE(current, n_words);

		if (has_same_attributes(current, prev, n_attributes))
		{
			/**
			 * It has the same attributes so it must be inconsistent,
			 * because we removed the duplicated lines before.
			 */
			inconsistency++;

			// Update max
			if (inconsistency > max_inconsistency)
			{
				max_inconsistency = inconsistency;
			}
		}
		else
		{
			// Differente attributes - reset JNSQ
			inconsistency = 0;
		}

		// Set the line JNSQ
		set_jnsq_bits(current, inconsistency, n_attributes, n_words,
					  n_bits_for_class);
	}

	return max_inconsistency;
}
