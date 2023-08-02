/*
 ============================================================================
 Name        : disjoint_matrix_mpi.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix
 ============================================================================
 */

#include "disjoint_matrix_mpi.h"

#include "dataset_hdf5.h"
#include "dataset_hdf5_mpi.h"
#include "disjoint_matrix.h"
#include "types/class_offsets_t.h"
#include "types/dataset_hdf5_t.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/oknok_t.h"
#include "types/word_t.h"
#include "utils/bit.h"
#include "utils/block.h"
#include "utils/math.h"
#include "utils/ranks.h"

#include "hdf5.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

oknok_t mpi_create_line_dataset(const dataset_hdf5_t* hdf5_dset,
								const dataset_t* dset, const dm_t* dm)
{
	/**
	 * Create line dataset
	 */
	hid_t dset_id = create_hdf5_dataset(hdf5_dset->file_id, DM_LINE_DATA,
										dm->n_matrix_lines, dset->n_words,
										H5T_NATIVE_UINT64);

	// Write dataset attributes
	herr_t err
		= write_dm_attributes(dset_id, dset->n_attributes, dm->n_matrix_lines);
	assert(err != NOK);

	// Allocate output buffer
	word_t* buffer
		= (word_t*) malloc(N_LINES_OUT * dset->n_words * sizeof(word_t));
	assert(buffer != NULL);

	// Current output line index
	uint32_t offset = dm->s_offset;

	uint32_t nc	   = dset->n_classes;
	uint32_t nobs  = dset->n_observations;
	uint32_t* nopc = dset->n_observations_per_class;
	word_t** opc   = dset->observations_per_class;

	// DO IT
	/**
	 * Current line being generated
	 */
	uint32_t cl = 1;

	/**
	 * Fill class offsets for first step of the computation
	 */
	class_offsets_t class_offsets;
	calculate_initial_offsets(dset, dm->s_offset, &class_offsets);
	uint32_t ca = class_offsets.classA;
	uint32_t ia = class_offsets.indexA;
	uint32_t cb = class_offsets.classB;
	uint32_t ib = class_offsets.indexB;

	// Current buffer line
	word_t* bl = buffer;

	// Number of lines currently on the buffer
	uint8_t n_lines_out = 0;

	while (ca < nc - 1)
	{
		word_t** bla = opc + ca * nobs;

		while (ia < nopc[ca])
		{
			word_t* la = *(bla + ia);

			while (cb < nc)
			{
				word_t** blb = opc + cb * nobs;

				while (ib < nopc[cb])
				{
					if (cl > dm->s_size)
					{
						goto done;
					}

					// Build one line
					word_t* lb = *(blb + ib);

					for (uint32_t n = 0; n < dset->n_words; n++)
					{
						(*bl) = la[n] ^ lb[n];
						bl++;
					}

					n_lines_out++;
					if (n_lines_out == N_LINES_OUT)
					{
						write_n_lines(dset_id, offset, n_lines_out,
									  dset->n_words, H5T_NATIVE_UINT64, buffer);

						offset += n_lines_out;
						n_lines_out = 0;
						bl			= buffer;
					}

					ib++;
					cl++;
				}
				cb++;
				ib = 0;
			}
			ia++;
			cb = ca + 1;
			ib = 0;
		}
		ca++;
		ia = 0;
		cb = ca + 1;
		ib = 0;
	}

done:
	// We may have lines to write
	if (n_lines_out > 0)
	{
		write_n_lines(dset_id, offset, n_lines_out, dset->n_words,
					  H5T_NATIVE_UINT64, buffer);
	}

	free(buffer);

	H5Dclose(dset_id);

	return OK;
}

oknok_t mpi_create_column_dataset(const dataset_hdf5_t* hdf5_dset,
								  const dataset_t* dset, const dm_t* dm,
								  const int rank, const int size)
{

	// Number of words in a line ON OUTPUT DATASET
	uint32_t out_n_words = dm->n_matrix_lines / WORD_BITS
		+ (dm->n_matrix_lines % WORD_BITS != 0);

	/**
	 * CREATE OUTPUT DATASET
	 *
	 * All processes must participate in the dataset creation event
	 */

	hid_t dset_id = create_hdf5_dataset(hdf5_dset->file_id, DM_COLUMN_DATA,
										dset->n_attributes, out_n_words,
										H5T_NATIVE_UINT64);

	/**
	 * Create dataset to hold the totals
	 */
	hid_t totals_dset_id
		= create_hdf5_dataset(hdf5_dset->file_id, DM_ATTRIBUTE_TOTALS, 1,
							  dset->n_attributes, H5T_NATIVE_UINT32);

	/**
	 * Attribute totals buffer
	 */
	uint32_t* attr_buffer = NULL;

	// We split the atributes to each process by word block
	// This process will process this many words from the input dataset
	uint32_t n_words_to_process = BLOCK_SIZE(rank, size, dset->n_words);

	// If we don't have anything to write return here
	if (n_words_to_process == 0)
	{
		H5Dclose(dset_id);
		H5Dclose(totals_dset_id);
		return OK;
	}

	// The attribute blocks to generate/save start at
	uint32_t attribute_block_start = BLOCK_LOW(rank, size, dset->n_words);
	uint32_t attribute_block_end   = attribute_block_start + n_words_to_process;

	/**
	 * Allocate input buffer
	 *
	 * Input buffer will hold one word per disjoint matrix line
	 * Rounding to nearest multiple of 64 so we don't have to worry when
	 * transposing the last lines
	 */
	word_t* in_buffer = (word_t*) malloc(out_n_words * 64 * sizeof(word_t));

	/**
	 * Allocate output buffer
	 *
	 * Will hold up the matrix lines of up to 64 attributes
	 */
	word_t* out_buffer = (word_t*) malloc(out_n_words * 64 * sizeof(word_t));

	// Start of the lines block to transpose
	word_t* transpose_index = NULL;

	/**
	 * Allocate attribute totals buffer
	 * We save the totals for each attribute.
	 * This saves time when selecting the first best attribute
	 */
	attr_buffer = (uint32_t*) calloc(dset->n_attributes, sizeof(uint32_t));

	uint32_t n_remaining_lines_to_write = n_words_to_process * 64;
	if (attribute_block_end * 64 > dset->n_attributes)
	{
		n_remaining_lines_to_write
			= dset->n_attributes - (attribute_block_start * 64);
	}

	uint32_t n_lines_to_write = 64;

	for (uint32_t current_attribute_word = attribute_block_start;
		 current_attribute_word < attribute_block_end;
		 current_attribute_word++, n_remaining_lines_to_write -= 64)
	{

		if (n_remaining_lines_to_write < 64)
		{
			n_lines_to_write = n_remaining_lines_to_write;
		}

		/**
		 * !TODO: Confirm that generating the column is faster than
		 * reading it from the line dataset
		 */
		generate_dm_column(dset, current_attribute_word, in_buffer);

		transpose_index = in_buffer;

		/**
		 * Transpose all lines but the last one
		 * The last word may not may not have 64 attributes
		 * and we're working with garbage, but it's fine because
		 * we trim them after the main loop
		 */
		uint32_t ow = 0;
		for (ow = 0; ow < out_n_words - 1; ow++, transpose_index += 64)
		{
			/**
			 * Transpose a 64x64 block in place
			 */
			// Transpose
			transpose64(transpose_index);

			// Append to output buffer
			for (uint8_t l = 0; l < n_lines_to_write; l++)
			{
				out_buffer[l * out_n_words + ow] = transpose_index[l];

				// Update attribute totals
				attr_buffer[(current_attribute_word - attribute_block_start)
								* WORD_BITS
							+ l]
					+= __builtin_popcountl(transpose_index[l]);
			}
		}

		// Last word

		// Transpose
		transpose64(transpose_index);

		// If it's last word we may have to mask the last bits that are noise
		word_t n_bits_to_check_mask = 0xffffffffffffffff
			<< (out_n_words * 64 - dm->n_matrix_lines);

		// Append to output buffer
		for (uint8_t l = 0; l < n_lines_to_write; l++)
		{
			transpose_index[l] &= n_bits_to_check_mask;

			out_buffer[l * out_n_words + ow] = transpose_index[l];

			// Update attribute totals
			attr_buffer[(current_attribute_word - attribute_block_start)
							* WORD_BITS
						+ l]
				+= __builtin_popcountl(transpose_index[l]);
		}

		// Save transposed array to file
		write_n_lines(dset_id, current_attribute_word * 64, n_lines_to_write,
					  out_n_words, H5T_NATIVE_UINT64, out_buffer);
	}

	free(out_buffer);
	free(in_buffer);

	H5Dclose(dset_id);

	/**
	 * Remove extra values
	 * The number of attributes may not be divisable by 64, so we could have
	 * extra values in the attr_buffer that are not necessary.
	 */
	uint32_t n_attributes = n_words_to_process * WORD_BITS;
	if (attribute_block_end * WORD_BITS > dset->n_attributes)
	{
		n_attributes = dset->n_attributes - attribute_block_start * WORD_BITS;
	}

	mpi_write_attribute_totals(totals_dset_id,
							   attribute_block_start * WORD_BITS, n_attributes,
							   attr_buffer);

	free(attr_buffer);
	H5Dclose(totals_dset_id);

	return OK;
}

oknok_t mpi_write_attribute_totals(const hid_t dataset_id, const uint32_t start,
								   const uint32_t n_attributes,
								   const uint32_t* data)
{

	hsize_t offset[2] = { 0, start };
	hsize_t count[2]  = { 1, n_attributes };

	write_to_hdf5_dataset(dataset_id, offset, count, H5T_NATIVE_UINT32, data);

	return OK;
}

oknok_t calculate_initial_offsets(const dataset_t* dataset, const uint32_t line,
								  class_offsets_t* class_offsets)
{

	/**
	 * Number of classes in dataset
	 */
	uint32_t nc = dataset->n_classes;

	/**
	 * Number of observations per class in dataset
	 */
	uint32_t* nopc = dataset->n_observations_per_class;

	/**
	 * Calculate the conditions for the first element of the disjoint matrix
	 * for this process. We always process the matrix in sequence, so this
	 * data is enough.
	 */
	if (nc == 2)
	{
		// For 2 classes the calculation is direct
		// This process will start working from here
		class_offsets->classA = 0;
		class_offsets->indexA = line / nopc[1];
		class_offsets->classB = 1;
		class_offsets->indexB = line % nopc[1];
	}
	else
	{
		uint32_t cl = 0;
		uint32_t ca = 0;
		uint32_t ia = 0;
		uint32_t cb = 0;
		uint32_t ib = 0;

		// I still haven't found a better way for more than 2 classes...
		// TODO: is there a better way?
		for (ca = 0; ca < nc - 1; ca++)
		{
			for (ia = 0; ia < nopc[ca]; ia++)
			{
				for (cb = ca + 1; cb < nc; cb++)
				{
					for (ib = 0; ib < nopc[cb]; ib++, cl++)
					{
						if (cl == line)
						{
							// This process will start working from here
							class_offsets->classA = ca;
							class_offsets->indexA = ia;
							class_offsets->classB = cb;
							class_offsets->indexB = ib;

							return OK;
						}
					}
				}
			}
		}
	}

	return OK;
}
