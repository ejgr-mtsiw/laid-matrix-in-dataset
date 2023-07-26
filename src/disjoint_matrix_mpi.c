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
#include "types/dataset_hdf5_t.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/oknok_t.h"
#include "types/steps_t.h"
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
		= H5Dcreate2(hdf5_dset->file_id, DM_LINE_DATA, H5T_NATIVE_ULONG,
					 filespace_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

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

	// Calculate how many and which lines this process will generate
	steps_t* s	   = dm->steps + dm->s_offset;
	steps_t* s_end = s + dm->s_size;

	// Allocate line totals buffer. For each line we store the number of bits
	// set
	uint32_t* lt_buffer = (uint32_t*) calloc(dm->s_size, sizeof(uint32_t));
	assert(lt_buffer != NULL);

	// Current line index
	uint32_t offset = dm->s_offset;

	// Current line index on line totals buffer
	uint32_t clt = 0;

	while (s < s_end)
	{
		uint8_t n_lines_out = 0;

		// Current buffer line
		word_t* bl = buffer;

		for (n_lines_out = 0; n_lines_out < N_LINES_OUT && s < s_end;
			 n_lines_out++, s++)
		{
			// Build one line
			word_t* la = dset->data + (s->indexA * dset->n_words);
			word_t* lb = dset->data + (s->indexB * dset->n_words);

			for (uint32_t n = 0; n < dset->n_words; n++)
			{
				bl[n] = la[n] ^ lb[n];

				// Update line total
				lt_buffer[clt] += __builtin_popcountl(bl[n]);
			}

			bl += dset->n_words;
			clt++;
		}

		// Save lines to file
		write_n_lines(dset_id, offset, n_lines_out, dset->n_words, buffer);

		offset += n_lines_out;
	}

	H5Dclose(dset_id);

	free(buffer);

	// Save line totals dataset
	mpi_write_line_totals(hdf5_dset, dm, lt_buffer);

	free(lt_buffer);

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
	err = H5Dwrite(dset_id, H5T_NATIVE_ULONG, memspace_id, filespace_id,
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

	// Round to nearest 64 so we don't have to worry when transposing
	uint32_t out_n_lines = roundUp(dset->n_attributes, WORD_BITS);

	// CREATE OUTPUT DATASET
	// Output dataset dimensions
	hsize_t dimensions[2] = { out_n_lines, out_n_words };

	hid_t filespace_id = H5Screate_simple(2, dimensions, NULL);
	assert(filespace_id != NOK);

	// Create a dataset creation property list
	hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	assert(dcpl_id != NOK);

	/*
	H5Pset_layout(dcpl_id, H5D_CHUNKED);

	The choice of the chunk size affects performance!
	for now we will choose one line
	hsize_t chunk_dimensions[2] = { 1, n_words };

	H5Pset_chunk(dcpl_id, 2, chunk_dimensions);
*/

	// Create a dataset access property list
	hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
	assert(dapl_id != NOK);

	/*
	H5Pset_chunk_cache(dapl_id, H5D_CHUNK_CACHE_NSLOTS_DEFAULT,
	1000*n_words*8, 1);
*/

	// Create the dataset collectively
	hid_t dset_id
		= H5Dcreate2(hdf5_dset->file_id, DM_COLUMN_DATA, H5T_NATIVE_ULONG,
					 filespace_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	// Close resources
	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);
	H5Sclose(filespace_id);

	/**
	 * Allocate input buffer
	 * Rounding to nearest multiple of 64 so we don't have to worry when
	 * transposing the last lines
	 */
	word_t* in_buffer
		= (word_t*) calloc(out_n_words * WORD_BITS, sizeof(word_t));

	// Allocate output buffer
	word_t* out_buffer
		= (word_t*) calloc(out_n_words * WORD_BITS, sizeof(word_t));

	// Allocate transpose buffer
	word_t* t_buffer = calloc(WORD_BITS, sizeof(word_t));

	// This process will process this many words from the input dataset
	uint32_t n_words_to_process = BLOCK_SIZE(rank, size, dset->n_words);

	// The blocks to generate/save start at
	uint32_t start = BLOCK_LOW(rank, size, dset->n_words);
	uint32_t end   = start + n_words_to_process;

	/**
	 * Allocate attribute totals buffer
	 * We save the totals for each attribute.
	 * This saves time when selecting the first best attribute
	 */
	uint32_t* attr_buffer
		= (uint32_t*) calloc(n_words_to_process * WORD_BITS, sizeof(uint32_t));

	for (uint32_t iw = start; iw < end; iw++)
	{
		/**
		 * !TODO: Confirm that generating the column is faster than
		 * reading it from the line dataset
		 */
		generate_dm_column(dset, dm, iw, in_buffer);

		/**
		 * Transpose lines
		 */
		for (uint32_t ow = 0; ow < out_n_words; ow++)
		{
			/**
			 * Read 64x64 bits block from input buffer
			 * The last word may not may not have 64 attributes
			 * and we're working with garbage, but it's fine as
			 * long as we don't lose count of n_attributes
			 */
			memcpy(t_buffer, in_buffer + (ow * WORD_BITS),
				   sizeof(word_t) * WORD_BITS);

			// Transpose
			transpose64(t_buffer);

			// Append to output buffer
			for (uint8_t l = 0; l < WORD_BITS; l++)
			{
				out_buffer[l * out_n_words + ow] = t_buffer[l];

				// Update attribute totals
				attr_buffer[(iw - start) * WORD_BITS + l]
					+= __builtin_popcountl(t_buffer[l]);
			}
		}

		// Save transposed array to file

		// Lets try and save 64 full lines at once!
		hsize_t offset[2] = { iw * WORD_BITS, 0 };
		hsize_t count[2]  = { WORD_BITS, out_n_words };

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

		// We're writing 64 lines at once
		hsize_t mem_dimensions[2] = { WORD_BITS, out_n_words };

		// Create a memory dataspace to indicate the size of our buffer/chunk
		hid_t memspace_id = H5Screate_simple(2, mem_dimensions, NULL);
		assert(memspace_id != NOK);

		// set up the collective transfer properties list
		hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
		assert(xfer_plist != NOK);

		H5Dwrite(dset_id, H5T_NATIVE_ULONG, memspace_id, filespace_id,
				 xfer_plist, out_buffer);

		H5Pclose(xfer_plist);
		H5Sclose(memspace_id);
		H5Sclose(filespace_id);
	}

	free(out_buffer);
	free(t_buffer);
	free(in_buffer);

	H5Dclose(dset_id);

	/**
	 * Remove extra values
	 * The number of attributes may not be divisable by 64, so we could have
	 * extra values in the attr_buffer that are not necessary.
	 */
	uint32_t n_lines = n_words_to_process * WORD_BITS;
	if ((start + n_words_to_process) * WORD_BITS > dset->n_attributes)
	{
		n_lines = dset->n_attributes - start * WORD_BITS;
	}

	if (n_lines > 0)
	{
		mpi_write_attribute_totals(hdf5_dset, attr_buffer, start * WORD_BITS,
								   n_lines, dset->n_attributes);
	}

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
		= H5Dcreate2(hdf5_dset->file_id, DM_LINE_TOTALS, H5T_NATIVE_UINT32,
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
		= H5Dcreate2(hdf5_dset->file_id, DM_ATTRIBUTE_TOTALS, H5T_NATIVE_UINT32,
					 filespace_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);

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
