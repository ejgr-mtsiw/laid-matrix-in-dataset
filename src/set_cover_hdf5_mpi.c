/*
 ============================================================================
 Name        : set_cover.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to apply the set cover algorithm
 ============================================================================
 */

#include "set_cover_hdf5_mpi.h"

#include "dataset_hdf5.h"
#include "set_cover.h"
#include "types/cover_t.h"
#include "types/dataset_hdf5_t.h"
#include "types/dm_t.h"
#include "types/oknok_t.h"
#include "types/word_t.h"
#include "utils/bit.h"

#include "hdf5.h"

#include <stdint.h>
#include <stdlib.h>

oknok_t get_column(hid_t dataset_id, uint32_t attribute, uint32_t start,
				   uint32_t n_words, word_t* column)
{
	/**
	 * Setup offset
	 */
	hsize_t offset[2] = { attribute, start };

	/**
	 * Setup count
	 */
	hsize_t count[2] = { 1, n_words };

	const hsize_t dimensions[1] = { n_words };

	/**
	 * Create a memory dataspace to indicate the size of our buffer/chunk
	 */
	hid_t memspace_id = H5Screate_simple(1, dimensions, NULL);

	/**
	 * Setup line dataspace
	 */
	hid_t dataspace_id = H5Dget_space(dataset_id);

	// Select hyperslab on file dataset
	H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count,
						NULL);

	// Read line from dataset
	H5Dread(dataset_id, H5T_NATIVE_ULONG, memspace_id, dataspace_id,
			H5P_DEFAULT, column);

	H5Sclose(dataspace_id);
	H5Sclose(memspace_id);

	return OK;
}

oknok_t update_covered_lines_mpi(const word_t* column, const uint32_t n_words,
								 word_t* covered_lines)
{
	for (uint32_t w = 0; w < n_words; w++)
	{
		BITMASK_SET(covered_lines[w], column[w]);
	}

	return OK;
}

oknok_t update_attribute_totals_mpi(cover_t* cover,
									dataset_hdf5_t* line_dataset,
									word_t* column)
{
	oknok_t ret = OK;

	word_t* line = (word_t*) malloc(sizeof(word_t) * cover->n_words_in_a_line);
	if (line == NULL)
	{
		fprintf(stderr, "Error allocating line buffer.\n");
		return NOK;
	}

	/**
	 * Define the lines this process must process
	 */
	uint32_t current_line = cover->column_offset_words * WORD_BITS;
	uint32_t end_line	  = current_line + cover->column_n_words * WORD_BITS;
	if (end_line > cover->n_matrix_lines)
	{
		end_line = cover->n_matrix_lines;
	}

	for (uint32_t w = 0; w < cover->column_n_words; w++)
	{
		/**
		 * The column corresponds to the line values of the best attribute
		 */
		word_t lines = column[w];

		// Ignore lines already covered
		lines &= ~cover->covered_lines[w];

		// Check the remaining lines
		for (int8_t bit = WORD_BITS - 1; bit >= 0 && current_line < end_line;
			 bit--, current_line++)
		{
			if (lines & AND_MASK_TABLE[bit])
			{
				// This line is covered by the best attribute

				// Read line from dataset
				ret = hdf5_read_line(line_dataset, current_line,
									 cover->n_words_in_a_line, line);
				if (ret != OK)
				{
					goto out_free_line_buffer;
				}

				// Increment totals
				add_line_contribution(cover, line);
			}
		}
	}

out_free_line_buffer:
	free(line);

	return ret;
}
