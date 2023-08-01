/*
 ============================================================================
 Name        : disjoint_matrix_mpi.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix
 ============================================================================
 */

#include "disjoint_matrix_mpi.h"

#include "dataset_hdf5.h"
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
	// Dataset dimensions
	hsize_t dimensions[2] = { dm->n_matrix_lines, dset->n_words };

	hid_t filespace_id = H5Screate_simple(2, dimensions, NULL);
	assert(filespace_id != NOK);

	// Create a dataset creation property list
	hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	assert(dcpl_id != NOK);

	// Create a dataset access property list
	hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
	assert(dapl_id != NOK);

	// Create the dataset collectively
	hid_t dset_id
		= H5Dcreate(hdf5_dset->file_id, DM_LINE_DATA, H5T_NATIVE_UINT64,
					filespace_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	// Close resources
	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);
	H5Sclose(filespace_id);

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
									  dset->n_words, buffer);

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
		write_n_lines(dset_id, offset, n_lines_out, dset->n_words, buffer);
	}

	free(buffer);

	H5Dclose(dset_id);

	return OK;
}

oknok_t write_n_lines(hid_t dset_id, uint32_t start, uint8_t n_lines_out,
					  uint32_t n_words, word_t* buffer)
{
	/**
	 * Setup dataspace
	 * If writing to a portion of a dataset in a loop, be sure
	 * to close the dataspace with each iteration, as this
	 * can cause a large temporary "memory leak".
	 * "Achieving High Performance I/O with HDF5"
	 */
	hid_t filespace_id = H5Dget_space(dset_id);
	assert(filespace_id != NOK);

	// We will write n_lines_out lines at a time
	hsize_t count[2]  = { n_lines_out, n_words };
	hsize_t offset[2] = { start, 0 };

	// Select hyperslab on file dataset
	herr_t err = H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL,
									 count, NULL);
	assert(err != NOK);

	// Create a memory dataspace to indicate the size of our buffer
	hsize_t mem_dimensions[2] = { n_lines_out, n_words };
	hid_t memspace_id		  = H5Screate_simple(2, mem_dimensions, NULL);
	assert(memspace_id != NOK);

	// set up the collective transfer properties list
	hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
	assert(xfer_plist != NOK);

	/**
	 * H5FD_MPIO_COLLECTIVE transfer mode is not favourable:
	 * https://docs.hdfgroup.org/hdf5/rfc/coll_ind_dd6.pdf
	 */
	/*
	err = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
	assert(err != NOK);
	*/

	// Write buffer to dataset
	err = H5Dwrite(dset_id, H5T_NATIVE_UINT64, memspace_id, filespace_id,
				   xfer_plist, buffer);
	assert(err != NOK);

	H5Pclose(xfer_plist);
	H5Sclose(memspace_id);
	H5Sclose(filespace_id);
	return OK;
}

oknok_t mpi_create_column_dataset(const dataset_hdf5_t* hdf5_dset,
								  const dataset_t* dset, const dm_t* dm,
								  const int rank, const int size)
{

	// Number of words in a line ON OUTPUT DATASET
	uint32_t out_n_words = dm->n_matrix_lines / WORD_BITS
		+ (dm->n_matrix_lines % WORD_BITS != 0);

	// CREATE OUTPUT DATASET
	// Output dataset dimensions
	hsize_t dimensions[2] = { dset->n_attributes, out_n_words };

	hid_t filespace_id = H5Screate_simple(2, dimensions, NULL);
	assert(filespace_id != NOK);

	// Create a dataset creation property list
	hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	assert(dcpl_id != NOK);

	// Create a dataset access property list
	hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
	assert(dapl_id != NOK);

	// Create the dataset collectively
	hid_t dset_id
		= H5Dcreate(hdf5_dset->file_id, DM_COLUMN_DATA, H5T_NATIVE_UINT64,
					filespace_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	// Close resources
	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);
	H5Sclose(filespace_id);

	/**
	 * Attribute totals buffer
	 */
	uint32_t* attr_buffer = NULL;

	// We split the atributes to each process by word block
	// This process will process this many words from the input dataset
	uint32_t n_words_to_process = BLOCK_SIZE(rank, size, dset->n_words);

	// The attribute blocks to generate/save start at
	uint32_t attribute_block_start = BLOCK_LOW(rank, size, dset->n_words);
	uint32_t attribute_block_end   = attribute_block_start + n_words_to_process;

	// If we don't have anything to write return here
	if (n_words_to_process == 0)
	{
		goto done;
	}

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
	 * Will hold up the matrix lines of to 64 attributes
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
		 * Transpose lines
		 */
		uint32_t ow = 0;
		for (ow = 0; ow < out_n_words - 1; ow++, transpose_index += 64)
		{
			/**
			 * Read 64x64 bits block from input buffer
			 * The last word may not may not have 64 attributes
			 * and we're working with garbage, but it's fine as
			 * long as we don't lose count of n_attributes
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
		// Lets try and save up to 64 full lines at once!
		hsize_t offset[2] = { current_attribute_word * 64, 0 };
		hsize_t count[2]  = { n_lines_to_write, out_n_words };

		/**
		 * Setup dataspace
		 * If writing to a portion of a dataset in a loop, be sure
		 * to close the dataspace with each iteration, as this
		 * can cause a large temporary "memory leak".
		 * "Achieving High Performance I/O with HDF5"
		 */
		filespace_id = H5Dget_space(dset_id);

		H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL, count,
							NULL);

		// We're writing `n_lines` lines at once
		hsize_t mem_dimensions[2] = { n_lines_to_write, out_n_words };

		// Create a memory dataspace to indicate the size of our buffer/chunk
		hid_t memspace_id = H5Screate_simple(2, mem_dimensions, NULL);
		assert(memspace_id != NOK);

		// set up the collective transfer properties list
		hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
		assert(xfer_plist != NOK);

		H5Dwrite(dset_id, H5T_NATIVE_UINT64, memspace_id, filespace_id,
				 xfer_plist, out_buffer);

		H5Pclose(xfer_plist);
		H5Sclose(memspace_id);
		H5Sclose(filespace_id);
	}

	free(out_buffer);
	free(in_buffer);

done:
	H5Dclose(dset_id);

	/**
	 * Remove extra values
	 * The number of attributes may not be divisable by 64, so we could have
	 * extra values in the attr_buffer that are not necessary.
	 */
	uint32_t n_attributes = n_words_to_process * WORD_BITS;
	if ((attribute_block_start + n_words_to_process) * WORD_BITS
		> dset->n_attributes)
	{
		n_attributes = dset->n_attributes - attribute_block_start * WORD_BITS;
	}

	/**
	 * All processes must go here, even if we don't have nothing to write
	 * because there's a dataset creation event inside!
	 */
	mpi_write_attribute_totals(hdf5_dset, attr_buffer,
							   attribute_block_start * WORD_BITS, n_attributes,
							   dset->n_attributes);

	free(attr_buffer);

	return OK;
}

oknok_t mpi_write_line_totals(const dataset_hdf5_t* hdf5_dset, const dm_t* dm,
							  const uint32_t* data)
{
	// Output dataset dimensions
	hsize_t dimensions[2] = { dm->n_matrix_lines, 1 };

	hid_t filespace_id = H5Screate_simple(2, dimensions, NULL);
	assert(filespace_id != NOK);

	// Create a dataset creation property list
	hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	assert(dcpl_id != NOK);

	// Create a dataset access property list
	hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
	assert(dapl_id != NOK);

	// Create the dataset collectively
	hid_t dset_id
		= H5Dcreate(hdf5_dset->file_id, DM_LINE_TOTALS, H5T_NATIVE_UINT32,
					filespace_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);

	hsize_t offset[2] = { dm->s_offset, 0 };
	hsize_t count[2]  = { dm->s_size, 1 };

	// Create a memory dataspace to indicate the size of our buffer/chunk
	hsize_t mem_dimensions[2] = { dm->s_size, 1 };
	hid_t memspace_id		  = H5Screate_simple(2, mem_dimensions, NULL);
	assert(memspace_id != NOK);

	H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL, count,
						NULL);

	// set up the collective transfer properties list
	hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
	assert(xfer_plist != NOK);

	H5Dwrite(dset_id, H5T_NATIVE_UINT32, memspace_id, filespace_id, xfer_plist,
			 data);

	H5Pclose(xfer_plist);
	H5Sclose(memspace_id);
	H5Sclose(filespace_id);
	H5Dclose(dset_id);

	return OK;
}

oknok_t mpi_write_attribute_totals(const dataset_hdf5_t* hdf5_dset,
								   const uint32_t* data, const uint32_t start,
								   const uint32_t n_lines,
								   const uint32_t n_attributes)
{
	// Output dataset dimensions
	hsize_t dimensions[2] = { n_attributes, 1 };

	hid_t filespace_id = H5Screate_simple(2, dimensions, NULL);
	assert(filespace_id != NOK);

	// Create a dataset creation property list
	hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	assert(dcpl_id != NOK);

	// Create a dataset access property list
	hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
	assert(dapl_id != NOK);

	// Create the dataset collectively
	hid_t dset_id
		= H5Dcreate(hdf5_dset->file_id, DM_ATTRIBUTE_TOTALS, H5T_NATIVE_UINT32,
					filespace_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);

	// If we don't have nothing to write return here
	if (n_lines == 0)
	{
		H5Sclose(filespace_id);
		H5Dclose(dset_id);

		return OK;
	}

	hsize_t offset[2] = { start, 0 };
	hsize_t count[2]  = { n_lines, 1 };

	// Create a memory dataspace to indicate the size of our buffer/chunk
	hsize_t mem_dimensions[2] = { n_lines, 1 };
	hid_t memspace_id		  = H5Screate_simple(2, mem_dimensions, NULL);
	assert(memspace_id != NOK);

	H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL, count,
						NULL);

	// Set up the collective transfer properties list
	hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
	assert(xfer_plist != NOK);

	H5Dwrite(dset_id, H5T_NATIVE_UINT32, memspace_id, filespace_id, xfer_plist,
			 data);

	H5Pclose(xfer_plist);
	H5Sclose(memspace_id);
	H5Sclose(filespace_id);
	H5Dclose(dset_id);

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
	uint32_t cl = 0;
	uint32_t ca = 0;
	uint32_t ia = 0;
	uint32_t cb = 0;
	uint32_t ib = 0;

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
		// I still haven«t found a better way for more than 2 classes...
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
