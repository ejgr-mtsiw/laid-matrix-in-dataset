/*
 ============================================================================
 Name        : disjoint_matrix.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix
 ============================================================================
 */

#include "disjoint_matrix.h"

#include "bit_utils.h"
#include "dataset.h"
#include "hdf5_dataset.h"
#include "oknok_t.h"
#include "timing.h"
#include "word_t.h"

#include "hdf5.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

uint32_t get_dm_n_lines(const dataset_t* dataset)
{
	// Calculate number of lines for the matrix
	uint32_t n = 0;

	uint32_t n_classes	  = dataset->n_classes;
	uint32_t* n_class_obs = dataset->n_observations_per_class;

	for (uint32_t i = 0; i < n_classes - 1; i++)
	{
		for (uint32_t j = i + 1; j < n_classes; j++)
		{
			n += n_class_obs[i] * n_class_obs[j];
		}
	}

	return n;
}

bool is_matrix_created(const char* filename)
{
	return hdf5_file_has_dataset(filename, DM_LINE_DATA);
}

oknok_t create_disjoint_matrix(const char* filename, const dataset_t* dataset)
{
	oknok_t ret = OK;

	// Open file
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
	assert(file_id != NOK);

	SETUP_TIMING
	TICK;

	ret = create_line_dataset(file_id, dataset);
	assert(ret != NOK);

	fprintf(stdout, "\n");
	TOCK(stdout)

	TICK;

	ret = create_column_dataset(file_id, dataset);
	assert(ret != NOK);

	fprintf(stdout, "\n");
	TOCK(stdout)

	H5Fclose(file_id);

	return ret;
}

oknok_t create_line_dataset(const hid_t file_id, const dataset_t* dataset)
{
	// Number of longs in a line
	uint32_t n_words = dataset->n_words;

	// Number of observations
	uint32_t n_observations = dataset->n_observations;

	// Number of classes
	uint32_t n_classes = dataset->n_classes;

	// Dataset data
	word_t* data = dataset->data;

	// Observations per class
	uint32_t* class_obs = dataset->observations_per_class;

	// Number of observations per class
	uint32_t* n_class_obs = dataset->n_observations_per_class;

	uint32_t n_lines = get_dm_n_lines(dataset);

	oknok_t ret = OK;

	// Dataset dimensions
	hsize_t dm_dimensions[2] = { n_lines, n_words };

	hid_t dm_dataset_space_id = H5Screate_simple(2, dm_dimensions, NULL);
	if (dm_dataset_space_id < 0)
	{
		// Error creating file
		fprintf(stderr, "Error creating dataset space\n");
		return NOK;
	}

	// Create a dataset creation property list
	hid_t dm_property_list_id = H5Pcreate(H5P_DATASET_CREATE);
	// H5Pset_layout(dm_property_list_id, H5D_CHUNKED);

	// The choice of the chunk size affects performance!
	// for now we will choose one line
	// hsize_t dm_chunk_dimensions[2] = { 1, n_words };

	// H5Pset_chunk(dm_property_list_id, 2, dm_chunk_dimensions);

	// Create the dataset
	hid_t dm_dataset_id
		= H5Dcreate(file_id, DM_LINE_DATA, H5T_STD_U64LE, dm_dataset_space_id,
					H5P_DEFAULT, dm_property_list_id, H5P_DEFAULT);

	H5Sclose(dm_dataset_space_id);

	if (dm_dataset_id < 0)
	{
		fprintf(stderr, "Error creating disjoint matrix dataset\n");
		return NOK;
	}

	// Save attributes
	if (write_dm_attributes(dm_dataset_id, dataset->n_attributes, n_lines) < 0)
	{
		fprintf(stderr, "Error saving matrix atributes");
		ret = NOK;
		goto out_dataset;
	}

	// Close resources
	// H5Pclose(dm_property_list_id);

	hsize_t mem_dimensions[2] = { 1, n_words };
	// Create a memory dataspace to indicate the size of our buffer/chunk
	hid_t dm_memory_space_id = H5Screate_simple(2, mem_dimensions, NULL);
	if (dm_memory_space_id < 0)
	{
		fprintf(stderr, "Error creating disjoint matrix memory space\n");
		ret = NOK;
		goto out_memory_space;
	}

	// Allocate buffer
	word_t* buffer = (word_t*) malloc(sizeof(word_t) * n_words);

	// Allocate line totals buffer
	uint32_t* lt_buffer = (uint32_t*) calloc(n_lines, sizeof(uint32_t));

	// We will write one line at a time
	hsize_t count[2]  = { 1, n_words };
	hsize_t offset[2] = { 0, 0 };

	// Used to print out progress message
	uint32_t next_output = 0;

	// Current output line
	uint32_t c_line = 0;

	// Current line
	for (uint32_t i = 0; i < n_classes - 1; i++)
	{
		for (uint32_t j = i + 1; j < n_classes; j++)
		{
			for (uint32_t n_i = i * n_observations;
				 n_i < i * n_observations + n_class_obs[i]; n_i++)
			{
				for (uint32_t n_j = j * n_observations;
					 n_j < j * n_observations + n_class_obs[j]; n_j++, c_line++)
				{
					word_t* rowA = data + class_obs[n_i] * n_words;
					word_t* rowB = data + class_obs[n_j] * n_words;

					for (uint32_t n = 0; n < n_words; n++)
					{
						buffer[n] = rowA[n] ^ rowB[n];

						lt_buffer[c_line] += __builtin_popcountl(buffer[n]);
					}

					// Update offset
					offset[0] = c_line;

					// Setup dataspace
					// If writing to a portion of a dataset in a loop, be sure
					// to close the dataspace with each iteration, as this
					// can cause a large temporary "memory leak".
					// "Achieving High Performance I/O with HDF5"
					dm_dataset_space_id
						= H5Screate_simple(2, dm_dimensions, NULL);

					// Select hyperslab on file dataset
					H5Sselect_hyperslab(dm_dataset_space_id, H5S_SELECT_SET,
										offset, NULL, count, NULL);

					// Write buffer to dataset
					H5Dwrite(dm_dataset_id, H5T_NATIVE_ULONG,
							 dm_memory_space_id, dm_dataset_space_id,
							 H5P_DEFAULT, buffer);

					H5Sclose(dm_dataset_space_id);

					if (c_line > next_output)
					{
						fprintf(stdout,
								" - Writing disjoint matrix [1/2]: %0.0f%%  \r",
								((double) c_line) / n_lines * 100);
						fflush(stdout);

						next_output += n_lines / 10;
					}
				}
			}
		}
	}

	// Create line totals dataset
	herr_t err = write_line_totals(file_id, lt_buffer, n_lines);
	if (err < 0)
	{
		fprintf(stderr, "Error creating line totals dataset\n");
		ret = NOK;
	}

	free(lt_buffer);
	free(buffer);

out_memory_space:
	H5Sclose(dm_memory_space_id);

out_dataset:
	H5Dclose(dm_dataset_id);

	return ret;
}

oknok_t create_column_dataset(const hid_t file_id, const dataset_t* dataset)
{
	// Number of attributes
	uint32_t n_attributes = dataset->n_attributes;

	// Number of observations
	uint32_t n_observations = dataset->n_observations;

	// Number of classes
	uint32_t n_classes = dataset->n_classes;

	// Observations per class
	//uint32_t* class_obs = dataset->observations_per_class;

	// Number of observations per class
	uint32_t* n_class_obs = dataset->n_observations_per_class;

	// Number of words in a line FROM INPUT DATASET
	uint32_t in_n_words
		= n_attributes / WORD_BITS + (n_attributes % WORD_BITS != 0);

	// Number of lines from input dataset
	uint32_t in_n_lines = get_dm_n_lines(dataset);

	// Number of words in a line FROM OUTPUT DATASET
	uint32_t out_n_words
		= in_n_lines / WORD_BITS + (in_n_lines % WORD_BITS != 0);

	// Round to nearest 64 so we don't have to worry when transposing
	uint32_t out_n_lines = in_n_words * WORD_BITS;

	oknok_t ret = OK;

	// CREATE OUTPUT DATASET
	// Output dataset dimensions
	hsize_t out_dimensions[2] = { out_n_lines, out_n_words };

	hid_t out_dataspace_id = H5Screate_simple(2, out_dimensions, NULL);
	if (out_dataspace_id < 0)
	{
		// Error creating file
		fprintf(stderr, "Error creating dataset space\n");
		return NOK;
	}

	// Create a dataset creation property list
	hid_t out_property_list_id = H5Pcreate(H5P_DATASET_CREATE);
	// H5Pset_layout(out_property_list_id, H5D_CHUNKED);

	// The choice of the chunk size affects performance!
	// for now we will choose one line
	// hsize_t out_chunk_dimensions[2] = { 1, out_n_words };

	// H5Pset_chunk(out_property_list_id, 2, out_chunk_dimensions);

	// Create the dataset
	hid_t out_dataset_id
		= H5Dcreate(file_id, DM_COLUMN_DATA, H5T_STD_U64LE, out_dataspace_id,
					H5P_DEFAULT, out_property_list_id, H5P_DEFAULT);

	H5Sclose(out_dataspace_id);

	if (out_dataset_id < 0)
	{
		fprintf(stderr, "Error creating output dataset\n");
		return NOK;
	}

	// Close resources
	// H5Pclose(out_property_list_id);

	// We're writing 64 lines at once
	hsize_t out_mem_dimensions[2] = { WORD_BITS, out_n_words };

	// Create a memory dataspace to indicate the size of our buffer/chunk
	hid_t out_memspace_id = H5Screate_simple(2, out_mem_dimensions, NULL);
	if (out_memspace_id < 0)
	{
		fprintf(stderr, "Error creating disjoint matrix memory space\n");
		ret = NOK;
		goto out_out_memspace;
	}

	// Allocate input buffer
	// word_t *in_buffer = (word_t*) malloc(sizeof(word_t) * in_n_lines);
	// Rounding to nearest multiple of 64 so we don't have to worry when
	// transposing the last lines
	word_t* in_buffer
		= (word_t*) calloc(out_n_words * WORD_BITS, sizeof(word_t));

	// Allocate output buffer
	word_t* out_buffer
		= (word_t*) calloc(out_n_words * WORD_BITS, sizeof(word_t));

	// Allocate transpose buffer
	word_t* t_buffer = calloc(WORD_BITS, sizeof(word_t));

	// Allocate attribute totals buffer
	// Correct size
	// uint32_t *attribute_buffer = (uint32_t*) calloc(n_attributes,
	// sizeof(uint32_t));
	// Rounded to 64 bits
	uint32_t* attr_buffer = (uint32_t*) calloc(out_n_lines, sizeof(uint32_t));

	// Used to print out progress message
	uint32_t next_output = 0;

	word_t* current_buffer = in_buffer;

	for (uint32_t i = 0; i < in_n_words; i++)
	{
		current_buffer = in_buffer;
		for (uint32_t ci = 0; ci < n_classes - 1; ci++)
		{
			for (uint32_t cj = ci + 1; cj < n_classes; cj++)
			{
				for (uint32_t n_i = ci * n_observations;
					 n_i < ci * n_observations + n_class_obs[ci]; n_i++)
				{
					for (uint32_t n_j = cj * n_observations;
						 n_j < cj * n_observations + n_class_obs[cj];
						 n_j++, current_buffer++)
					{
						*current_buffer = 2;

						//						*current_buffer =
						// class_obs[n_i][i] ^ class_obs[n_j][i];
					}
				}
			}
		}

		// TRANSPOSE LINES
		for (uint32_t w = 0; w < out_n_words; w++)
		{
			// Read 64x64 bits block from input buffer
			//! WARNING: We may have fewer than 64 lines remaining
			// We may be reading garbage
			memcpy(t_buffer, in_buffer + (w * WORD_BITS),
				   sizeof(word_t) * WORD_BITS);

			// Transpose
			transpose64(t_buffer);

			// Append to output buffer
			for (uint8_t l = 0; l < WORD_BITS; l++)
			{
				out_buffer[l * out_n_words + w] = t_buffer[l];
			}
		}

		// Lets try and save 64 full lines at once!
		hsize_t out_offset[2] = { 0, 0 };
		hsize_t out_count[2]  = { WORD_BITS, out_n_words };

		// SAVE TRANSPOSED ARRAY
		out_offset[0] = i * WORD_BITS;

		// Setup dataspace
		// If writing to a portion of a dataset in a loop, be sure
		// to close the dataspace with each iteration, as this
		// can cause a large temporary "memory leak".
		// "Achieving High Performance I/O with HDF5"
		hid_t out_dataspace_id = H5Screate_simple(2, out_dimensions, NULL);

		H5Sselect_hyperslab(out_dataspace_id, H5S_SELECT_SET, out_offset, NULL,
							out_count, NULL);

		H5Dwrite(out_dataset_id, H5T_NATIVE_ULONG, out_memspace_id,
				 out_dataspace_id, H5P_DEFAULT, out_buffer);

		//		// Lets try and save 1 full line at once!
		//		hsize_t out_offset[2] = { 0, 0 };
		//		hsize_t out_count[2] = { 1, out_n_words };
		//
		//		// SAVE TRANSPOSED ARRAY
		//		for (uint8_t l = 0; l < WORD_BITS; l++) {
		//			out_offset[0] = i * WORD_BITS + l;
		//			H5Sselect_hyperslab(out_dataspace_id, H5S_SELECT_SET,
		// out_offset, 			NULL, out_count, NULL);
		//
		//			H5Dwrite(out_dataset_id, H5T_NATIVE_ULONG, out_memspace_id,
		//					out_dataspace_id, H5P_DEFAULT,
		//					out_buffer + out_n_words * l);
		//		}

		H5Sclose(out_dataspace_id);

		// Update attribute totals
		for (uint32_t at = 0; at < WORD_BITS; at++)
		{
			for (uint64_t l = at * out_n_words; l < (at + 1) * out_n_words; l++)
			{
				attr_buffer[out_offset[0] + at]
					+= __builtin_popcountl(out_buffer[l]);
			}
		}

		if (i > next_output)
		{
			fprintf(stdout, " - Writing disjoint matrix [2/2]: %0.0f%%      \r",
					((double) i) / in_n_words * 100);
			fflush(stdout);

			next_output += in_n_words / 10;
		}
	}

	// Create attribute totals dataset
	herr_t err = write_attribute_totals(file_id, attr_buffer, n_attributes);
	if (err < 0)
	{
		fprintf(stderr, "Error creating attribute totals dataset\n");
		ret = NOK;
	}

	free(t_buffer);
	free(attr_buffer);
	free(in_buffer);
	free(out_buffer);

out_out_memspace:
	H5Sclose(out_memspace_id);

	// out_out_dataset:
	H5Dclose(out_dataset_id);

	return ret;
}

herr_t write_dm_attributes(const hid_t dataset_id, const uint32_t n_attributes,
						   const uint32_t n_matrix_lines)
{
	herr_t ret = 0;

	ret = hdf5_write_attribute(dataset_id, N_ATTRIBUTES_ATTR, H5T_NATIVE_UINT,
							   &n_attributes);
	if (ret < 0)
	{
		return ret;
	}

	ret = hdf5_write_attribute(dataset_id, N_MATRIX_LINES_ATTR, H5T_NATIVE_UINT,
							   &n_matrix_lines);

	return ret;
}

herr_t write_line_totals(const hid_t file_id, const uint32_t* data,
						 const uint32_t n_lines)
{
	herr_t ret = 0;

	// Dataset dimensions
	hsize_t lt_dimensions[1] = { n_lines };

	hid_t lt_dataset_space_id = H5Screate_simple(1, lt_dimensions, NULL);
	if (lt_dataset_space_id < 0)
	{
		// Error creating file
		fprintf(stderr, "Error creating line totals dataset space\n");
		return lt_dataset_space_id;
	}

	// Create the dataset
	hid_t lt_dataset_id = H5Dcreate2(file_id, DM_LINE_TOTALS, H5T_STD_U32LE,
									 lt_dataset_space_id, H5P_DEFAULT,
									 H5P_DEFAULT, H5P_DEFAULT);
	if (lt_dataset_id < 0)
	{
		fprintf(stderr, "Error creating line totals dataset\n");
		ret = lt_dataset_id;
		goto out_lt_dataset_space;
	}

	herr_t err = H5Dwrite(lt_dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL,
						  H5P_DEFAULT, data);
	if (err < 0)
	{
		fprintf(stderr, "Error writing line totals\n");
		ret = err;
	}

out_lt_dataset_space:
	H5Sclose(lt_dataset_space_id);

	H5Dclose(lt_dataset_id);

	return ret;
}

herr_t write_attribute_totals(const hid_t file_id, const uint32_t* data,
							  const uint32_t n_attributes)
{
	herr_t ret = 0;

	// Dataset dimensions
	hsize_t dimensions[1] = { n_attributes };

	hid_t dataspace_id = H5Screate_simple(1, dimensions, NULL);
	if (dataspace_id < 0)
	{
		// Error creating file
		fprintf(stderr, "Error creating attribute totals dataset space\n");
		return dataspace_id;
	}

	// Create the dataset
	hid_t dataset_id
		= H5Dcreate2(file_id, DM_ATTRIBUTE_TOTALS, H5T_STD_U32LE, dataspace_id,
					 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dataset_id < 0)
	{
		fprintf(stderr, "Error creating attribute totals dataset\n");
		ret = dataset_id;
		goto out_dataspace;
	}

	herr_t err = H5Dwrite(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL,
						  H5P_DEFAULT, data);
	if (err < 0)
	{
		fprintf(stderr, "Error writing attribute totals\n");
		ret = err;
	}

out_dataspace:
	H5Sclose(dataspace_id);

	H5Dclose(dataset_id);

	return ret;
}
