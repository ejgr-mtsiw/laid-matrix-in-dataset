/*
 ============================================================================
 Name        : set_cover.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to apply the set cover algorithm
 ============================================================================
 */

#ifndef SET_COVER_H
#define SET_COVER_H

#include "types/cover_t.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/oknok_t.h"

#include "hdf5.h"

#include <stdint.h>

/**
 * reads initial attribute totals from metadata dataset
 */
oknok_t read_initial_attribute_totals(hid_t file_id,
									  uint32_t* attribute_totals);

/**
 * Searches the attribute totals array for the highest score and returns the
 * correspondent attribute index.
 * Returns -1 if there are no more attributes available.
 */
int64_t get_best_attribute_index(const uint32_t* totals,
								 const uint32_t n_attributes);

/**
 * Sets this attribute as selected
 */
oknok_t mark_attribute_as_selected(cover_t* cover, int64_t attribute);

/**
 * Updates the contribution of this line to the attributes totals
 */
oknok_t add_line_contribution(cover_t* cover, const word_t* line);

/**
 * Updates the list of covered lines, adding the lines covered by column
 */
oknok_t update_covered_lines(cover_t* cover, word_t* column);

/**
 * Calculates the initial totals for all attributes
 */
void calculate_initial_sum(const char* filename, const cover_t* cover);

/**
 * Prints the attributes that are part of the solution
 */
void print_solution(FILE* stream, cover_t* cover);

/**
 * Initializes (zeroes) the cover parameters
 */
void init_cover(cover_t* cover);

/**
 * Frees the allocated resources
 */
void free_cover(cover_t* cover);

#endif
