/*
 ============================================================================
 Name        : set_cover.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to apply the set cover algorithm
 ============================================================================
 */

#include "set_cover.h"

#include "dataset_hdf5.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/oknok_t.h"
#include "types/word_t.h"
#include "utils/bit.h"
#include "utils/timing.h"

#include "hdf5.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

oknok_t read_initial_attribute_totals(hid_t file_id, uint32_t* attribute_totals)
{
	// Open dataset
	hid_t dset_id = H5Dopen(file_id, DM_ATTRIBUTE_TOTALS, H5P_DEFAULT);
	assert(dset_id != NOK);

	// Read attribute totals
	herr_t status = H5Dread(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL,
							H5P_DEFAULT, attribute_totals);
	assert(status != NOK);

	status = H5Dclose(dset_id);
	assert(status != NOK);

	return OK;
}

int64_t get_best_attribute_index(const uint32_t* totals,
								 const uint32_t n_attributes)
{
	uint32_t max_total	  = 0;
	int64_t max_attribute = -1;

	for (uint32_t i = 0; i < n_attributes; i++)
	{
		if (totals[i] > max_total)
		{
			max_total	  = totals[i];
			max_attribute = i;
		}
	}

	return max_attribute;
}

oknok_t add_line_contribution(cover_t* cover, const word_t* line)
{
	/**
	 * Current attribute
	 */
	uint32_t c_attribute = 0;

	// Process words
	for (uint32_t c_word = 0; c_word < cover->n_words_in_a_line; c_word++)
	{
		for (int8_t bit = WORD_BITS - 1; bit >= 0; bit--, c_attribute++)
		{
			cover->attribute_totals[c_attribute]
				+= BIT_CHECK(line[c_word], bit);
		}
	}

	return OK;
}

oknok_t sub_line_contribution(cover_t* cover, const word_t* line)
{
	/**
	 * Current attribute
	 */
	uint32_t c_attribute = 0;

	// Process words
	for (uint32_t c_word = 0; c_word < cover->n_words_in_a_line; c_word++)
	{
		for (int8_t bit = WORD_BITS - 1; bit >= 0; bit--, c_attribute++)
		{
			cover->attribute_totals[c_attribute]
				-= BIT_CHECK(line[c_word], bit);
		}
	}

	return OK;
}

oknok_t update_covered_lines(cover_t* cover, word_t* column)
{
	for (uint32_t w = 0; w < cover->column_n_words; w++)
	{
		BITMASK_SET(cover->covered_lines[w], column[w]);
	}

	return OK;
}

oknok_t mark_attribute_as_selected(cover_t* cover, int64_t attribute)
{
	uint32_t attribute_word = attribute / WORD_BITS;
	uint8_t attribute_bit	= WORD_BITS - (attribute % WORD_BITS) - 1;

	BIT_SET(cover->selected_attributes[attribute_word], attribute_bit);

	return OK;
}

void print_solution(FILE* stream, cover_t* cover)
{
	fprintf(stream, "Solution: { ");

	uint32_t current_attribute = 0;
	uint32_t solution_size	   = 0;

	for (uint32_t w = 0; w < cover->n_words_in_a_line; w++)
	{
		for (int8_t bit = WORD_BITS - 1;
			 bit >= 0 && current_attribute < cover->n_attributes;
			 bit--, current_attribute++)
		{
			if (cover->selected_attributes[w] & AND_MASK_TABLE[bit])
			{
				// This attribute is set so it's part of the solution
				fprintf(stream, "%d ", current_attribute);
				solution_size++;
			}
		}
	}

	fprintf(stream, "}\nSolution has %d attributes: %d / %d = %3.4f%%\n",
			solution_size, solution_size, cover->n_attributes,
			((float) solution_size / (float) cover->n_attributes) * 100);
}

void free_cover(cover_t* cover)
{
	cover->n_attributes		 = 0;
	cover->n_matrix_lines	 = 0;
	cover->n_words_in_a_line = 0;
	cover->column_n_words	 = 0;

	free(cover->covered_lines);
	free(cover->selected_attributes);
	free(cover->attribute_totals);

	cover->covered_lines	   = NULL;
	cover->selected_attributes = NULL;
	cover->attribute_totals	   = NULL;
}

void init_cover(cover_t* cover)
{
	cover->n_attributes		   = 0;
	cover->n_matrix_lines	   = 0;
	cover->n_words_in_a_line   = 0;
	cover->column_n_words	   = 0;
	cover->covered_lines	   = NULL;
	cover->selected_attributes = NULL;
	cover->attribute_totals	   = NULL;
}
