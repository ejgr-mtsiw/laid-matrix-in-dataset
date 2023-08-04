/*
 ============================================================================
 Name        : types/cover_t.h
 Author      : Eduardo Ribeiro
 Description : Datatype representing a cover state
 ============================================================================
 */

#ifndef TYPES_COVER_T_H
#define TYPES_COVER_T_H

#include "types/word_t.h"

#include <stdint.h>

typedef struct cover_t
{
	/**
	 * Number of attributes
	 */
	uint32_t n_attributes;

	/**
	 * Total number of lines of the disjoint matrix
	 */
	uint32_t n_matrix_lines;

	/**
	 * Number of words needed to store a line
	 */
	uint32_t n_words_in_a_line;

	/**
	 * Offset of the disjoint matrix
	 */
	uint32_t column_offset_words;

	/**
	 * Number of words needed to store a column
	 */
	uint32_t column_n_words;

	/**
	 * Bit array of covered lines
	 */
	word_t* covered_lines;

	/**
	 * Number of lines that are not covered yet
	 */
	uint32_t n_uncovered_lines;

	/**
	 * Bit array of selected attributes
	 */
	word_t* selected_attributes;

	/**
	 * Array with the current totals for all attributes
	 */
	uint32_t* attribute_totals;
} cover_t;

#endif // TYPES_COVER_T_H
