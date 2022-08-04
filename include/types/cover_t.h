/*
 ============================================================================
 Name        : cover_t.h
 Author      : Eduardo Ribeiro
 Description : Datatype representing a cover state
 ============================================================================
 */

#ifndef COVER_T_H
#define COVER_T_H

#include "types/word_t.h"

#include <stdint.h>

typedef struct cover_t
{
	/**
	 * Number of attributes
	 */
	uint32_t n_attributes;

	/**
	 * Number of lines of the disjoint matrix
	 */
	uint32_t n_matrix_lines;

	/**
	 * Number of words needed to store a line
	 */
	uint32_t n_words_in_a_line;

	/**
	 * Number of words needed to store a column
	 */
	uint32_t n_words_in_a_column;

	/**
	 * Bit array of covered lines
	 */
	word_t* covered_lines;

	/**
	 * Bit array of selected attributes
	 */
	word_t* selected_attributes;

	/**
	 * Array with the current totals for all attributes
	 */
	uint32_t* attribute_totals;
} cover_t;

#endif // COVER_T_H
