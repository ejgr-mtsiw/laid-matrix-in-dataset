/*
 ============================================================================
 Name        : disjoint_matrix.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix
 ============================================================================
 */

#include "mpi_disjoint_matrix.h"

#include "bit_utils.h"
#include "disjoint_matrix.h"
#include "hdf5_dataset.h"
#include "oknok_t.h"
#include "word_t.h"

#include "hdf5.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// First element controlled by process id out of p processes, array length n
#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))

// Last element controlled by process id out of p processes, array length n
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id + 1), p, n) - 1)

// Size of the block controlled by process id out of p processes, array length n
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW((id + 1), p, n) - BLOCK_LOW(id, p, n))

// Process that controls item index from array with length n, p processes
#define BLOCK_OWNER(index, p, n) (((p) * ((index) + 1) - 1) / (n))

// oknok_t mpi_create_disjoint_matrix(const char* filename,
//								   const dataset_t* dataset, int mpi_rank,
//								   int mpi_size, MPI_Comm comm, MPI_Info info)
//{
//	oknok_t ret = OK;
//
//	// Just to stop warning/error
//	if (mpi_rank > 20)
//	{
//		printf("%d - %d\n", mpi_rank, mpi_size);
//	}
//
//	/* setup file access template */
//	hid_t acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
//	assert(acc_tpl != NOK);
//	/* set Parallel access with communicator */
//	herr_t err = H5Pset_fapl_mpio(acc_tpl, comm, info);
//	assert(err != NOK);
//
//	/* open the file collectively */
//	hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, acc_tpl);
//	assert(file_id != NOK);
//
//	/* Release file-access template */
//	ret = H5Pclose(acc_tpl);
//	assert(ret != NOK);
//
//	SETUP_TIMING
//	TICK;
//	ret = mpi_create_line_dataset(file_id, dataset, mpi_rank, mpi_size, comm);
//	assert(ret != NOK);
//
//	fprintf(stdout, "\n");
//	TOCK(stdout)
//
//	//	TICK;
//	//	ret = mpi_create_column_dataset(file_id, dataset, mpi_rank, mpi_size);
//	//	assert(ret != NOK);
//	//
//	//	fprintf(stdout, "\n");
//	//	TOCK(stdout)
//
//	MPI_Barrier(comm);
//
//	H5Fclose(file_id);
//
//	return ret;
// }

uint32_t roundUp(uint32_t numToRound, uint32_t multiple)
{
	if (multiple == 0)
		return numToRound;

	uint32_t remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;

	return numToRound + multiple - remainder;
}

oknok_t mpi_create_line_dataset(hdf5_dataset_t hdf5_dset, word_t* data,
								const uint32_t n_matrix_lines,
								const uint32_t n_words, steps_t* steps,
								const int rank, const int size)
{

	// Number of lines in the disjoint matrix
	// We're rounding up so we can write collectively size lines at a time
	// keeping them close so HDF5 can optimize the writing as a block
	uint32_t n_total_lines = roundUp(n_matrix_lines, size);

	// Dataset dimensions
	hsize_t file_dimensions[2] = { n_total_lines, n_words };

	hid_t file_space_id = H5Screate_simple(2, file_dimensions, NULL);
	assert(file_space_id != NOK);

	hid_t file_id = hdf5_dset.file_id;

	// Create a dataset creation property list
	hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	assert(dcpl_id != NOK);

	// H5Pset_layout(dcpl_id, H5D_CHUNKED);

	// The choice of the chunk size affects performance!
	// for now we will choose one line
	// hsize_t chunk_dimensions[2] = { 1, n_words };

	// H5Pset_chunk(dcpl_id, 2, chunk_dimensions);

	// Create a dataset access property list
	hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
	assert(dapl_id != NOK);

	// H5Pset_chunk_cache(dapl_id, H5D_CHUNK_CACHE_NSLOTS_DEFAULT,
	// 1000*n_words*8, 1);

	// Create the dataset collectively
	hid_t dset_id = H5Dcreate2(file_id, DM_LINE_DATA, H5T_STD_U64LE,
							   file_space_id, H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);
	// H5Sclose(file_space_id);

	// set up the collective transfer properties list
	hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
	assert(xfer_plist != NOK);

	herr_t err = 0;

	//	err = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
	//	assert(err != NOK);

	// Save attributes collectively
	//	err = write_dm_attributes(dset_id, dataset->n_attributes,
	// n_lines);
	// assert(err != NOK);

	hsize_t mem_dimensions[2] = { 1, n_words };
	// Create a memory dataspace to indicate the size of our buffer/chunk
	hid_t mem_space_id = H5Screate_simple(2, mem_dimensions, NULL);
	assert(mem_space_id != NOK);

	// Allocate buffer
	word_t* buffer = (word_t*) malloc(sizeof(word_t) * n_words);
	assert(buffer != NULL);

	// Allocate line totals buffer
	// uint32_t* lt_buffer = (uint32_t*) calloc(n_total_lines,
	// sizeof(uint32_t)); assert(lt_buffer != NULL);

	// This process will write this many lines
	uint32_t n_lines = BLOCK_SIZE(rank, size, n_total_lines);

	// The lines to generate/save start at
	uint32_t start = BLOCK_LOW(rank, size, n_total_lines);

	// We will write one line at a time
	hsize_t count[2]  = { 1, n_words };
	hsize_t offset[2] = { start, 0 };

	uint32_t end = start + n_lines;
	if (end > n_matrix_lines)
	{
		end = n_matrix_lines;
	}

	uint32_t c = start;

	//	for (; c < end; c++)
	//		{
	//			// Setup dataspace
	//			// If writing to a portion of a dataset in a loop, be sure
	//			// to close the dataspace with each iteration, as this
	//			// can cause a large temporary "memory leak".
	//			// "Achieving High Performance I/O with HDF5"
	//			file_space_id = H5Dget_space(dset_id);
	//			assert(file_space_id != NOK);
	//
	//			// We are writing collectively, so all processes must
	//participate even
	//			// if they don't have anything to write. So, if we're done with
	// our
	//			// share we must repeat unitl. everyone else finishes too.
	//			if (c < n_matrix_lines)
	//			{
	//				steps_t* s = steps + c;
	//
	//				word_t* la = data + (s->indexA * n_words);
	//				word_t* lb = data + (s->indexB * n_words);
	//
	//				for (uint32_t n = 0; n < n_words; n++)
	//				{
	//					buffer[n] = la[n] ^ lb[n];
	//				}
	//
	//				// Select hyperslab on file dataset
	//				herr_t err = H5Sselect_hyperslab(file_space_id,
	// H5S_SELECT_SET, 												 offset, NULL, count, NULL); 				assert(err
	// != NOK);
	//
	//				// printf("[%d] Wrote line %llu/%d\n", rank, offset[0],
	//				// n_total_lines);
	//			}
	//			else
	//			{
	//				// We're just going along for the write
	//				herr_t err = H5Sselect_none(file_space_id);
	//				assert(err != NOK);
	//
	//				H5Sselect_none(mem_space_id);
	//				assert(err != NOK);
	//
	//				//			printf("[%d] Wrote BLANK line %llu/%d\n", rank,
	//				//offset[0]+1, 				   n_total_lines);
	//			}
	//
	//			// Write buffer to dataset
	//			err = H5Dwrite(dset_id, H5T_NATIVE_ULONG, mem_space_id,
	// file_space_id, 						   xfer_plist, buffer); 			assert(err !=
	// NOK);
	//
	//			H5Sclose(file_space_id);
	//
	//			// Update offset
	//			offset[0] += size;
	//		}
	//	if (start + n_lines > n_matrix_lines)
		//	{
		//		end = start + n_lines;
		//	}
		//
		//	// Setup dataspace
		//	// If writing to a portion of a dataset in a loop, be sure
		//	// to close the dataspace with each iteration, as this
		//	// can cause a large temporary "memory leak".
		//	// "Achieving High Performance I/O with HDF5"
		//	file_space_id = H5Dget_space(dset_id);
		//	assert(file_space_id != NOK);
		//
		//	// We're just going along for the write
		//	err = H5Sselect_none(file_space_id);
		//	assert(err != NOK);
		//
		//	H5Sselect_none(mem_space_id);
		//	assert(err != NOK);
		//
		//	for (; c < end; c++)
		//	{
		//		// Write buffer to dataset
		//		err = H5Dwrite(dset_id, H5T_NATIVE_ULONG, mem_space_id,
		//file_space_id, 					   xfer_plist, buffer); 		assert(err != NOK);
		//
		//		// Update offset
		//		offset[0] += size;
		//	}

	for (; c < end; c++)
	{
		// Setup dataspace
		// If writing to a portion of a dataset in a loop, be sure
		// to close the dataspace with each iteration, as this
		// can cause a large temporary "memory leak".
		// "Achieving High Performance I/O with HDF5"
		file_space_id = H5Dget_space(dset_id);
		assert(file_space_id != NOK);

		steps_t* s = steps + c;

		word_t* la = data + (s->indexA * n_words);
		word_t* lb = data + (s->indexB * n_words);

		for (uint32_t n = 0; n < n_words; n++)
		{
			buffer[n] = la[n] ^ lb[n];
		}

		// Select hyperslab on file dataset
		err = H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, NULL,
								  count, NULL);
		assert(err != NOK);

		// Write buffer to dataset
		err = H5Dwrite(dset_id, H5T_NATIVE_ULONG, mem_space_id, file_space_id,
					   xfer_plist, buffer);
		assert(err != NOK);

		H5Sclose(file_space_id);

		// Update offset
		offset[0]++;
	}

	H5Dclose(dset_id);

	return OK;
}

// oknok_t mpi_create_line_dataset(const hid_t file_id, const dataset_t*
// dataset, 								const int mpi_rank, const int
// mpi_size,MPI_Comm comm)
//{
//	// Number of longs in a line
//	uint32_t n_words = dataset->n_words;
//
//	// Number of observations
//	uint32_t n_observations = dataset->n_observations;
//
//	// Number of classes
//	uint32_t n_classes = dataset->n_classes;
//
//	// Observations per class
//	word_t** observations_per_class = dataset->observations_per_class;
//
//	// Number of observations per class
//	uint32_t* n_observations_per_class = dataset->n_observations_per_class;
//
//	// Number of lines in the disjoint matrix
//	uint32_t n_total_lines = get_dm_n_lines(dataset);
//	n_total_lines = roundUp(n_total_lines, mpi_size);
//
//	// Dataset dimensions
//	hsize_t file_dimensions[2] = { n_total_lines, n_words };
//
//	hid_t file_space_id = H5Screate_simple(2, file_dimensions, NULL);
//	assert(file_space_id != NOK);
//
//	// Create a dataset creation property list
//	 //hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
//	 //assert(dcpl_id != NOK);
//
//	 //H5Pset_layout(dcpl_id, H5D_CHUNKED);
//
//	// The choice of the chunk size affects performance!
//	// for now we will choose one line
//	 //hsize_t chunk_dimensions[2] = { 1, n_words };
//
//	 //H5Pset_chunk(dcpl_id, 2, chunk_dimensions);
//
//	 // Create a dataset access property list
//	 		 //hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
//	 		 //assert(dapl_id != NOK);
//
//	 		 //H5Pset_chunk_cache(dapl_id, H5D_CHUNK_CACHE_NSLOTS_DEFAULT,
// 1000*n_words*8, 1);
//
//	// Create the dataset collectively
//	hid_t dset_id
//		= H5Dcreate2(file_id, DM_LINE_DATA, H5T_NATIVE_ULONG, file_space_id,
//					H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//	assert(dset_id != NOK);
//
//	//H5Pclose(dapl_id);
//	//H5Pclose(dcpl_id);
//	// H5Sclose(file_space_id);
//
//	//	file_space_id = H5Dget_space(dset_id);
//	//	assert(file_space_id != NOK);
//
//	 //set up the collective transfer properties list
//		hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
//		assert(xfer_plist != NOK);
//
//		herr_t err = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
//	  assert(err != NOK);
//
//	// Save attributes collectively
//	//	herr_t err = write_dm_attributes(dset_id, dataset->n_attributes,
//	// n_lines); 	assert(err != NOK);
//
//	hsize_t mem_dimensions[2] = { 1, n_words };
//	// Create a memory dataspace to indicate the size of our buffer/chunk
//	hid_t mem_space_id = H5Screate_simple(2, mem_dimensions, NULL);
//	assert(mem_space_id != NOK);
//
//	// Allocate buffer
//	word_t* buffer = (word_t*) malloc(sizeof(word_t) * n_words);
//	assert(buffer != NULL);
//
//	// Allocate line totals buffer
//	uint32_t* lt_buffer = (uint32_t*) calloc(n_total_lines, sizeof(uint32_t));
//	assert(lt_buffer != NULL);
//
//	// This process will write this many lines
//	uint32_t n_lines = BLOCK_SIZE(mpi_rank, mpi_size, n_total_lines);
//
//	// The lines to generate/save start at
//	uint32_t start = BLOCK_LOW(mpi_rank, mpi_size, n_total_lines);
//
//	// We will write one line at a time
//	hsize_t count[2]  = { 1, n_words };
//	hsize_t offset[2] = { mpi_rank, 0 };
//
//	uint32_t end = start+n_lines;
//
//	printf("start=%lu, end=%lu, n=%lu/%lu, count[]=(%lu,%lu), total
// datapoints=%lu\n", 		   (unsigned long) start, (unsigned long)
// end,(unsigned long) n_lines, 		   (unsigned long) n_total_lines,
// (unsigned long) count[0], 		   (unsigned long) count[1], (unsigned long)
// (n_lines * n_words));
//
//	// Used to print out progress message
//	// uint32_t next_output = 0;
//
//	// Current output line
//	uint32_t c_line = 0;
//
//	// Current line
//	for (uint32_t i = 0; i < n_classes - 1; i++)
//	{
//		for (uint32_t j = i + 1; j < n_classes; j++)
//		{
//			for (uint32_t n_i = i * n_observations;
//				 n_i < i * n_observations + n_observations_per_class[i]; n_i++)
//			{
//				for (uint32_t n_j = j * n_observations;
//					 n_j < j * n_observations + n_observations_per_class[j];
//					 n_j++, c_line++)
//				{
//					if (c_line < start)
//					{
//						continue;
//					}
//
//					if (c_line == end)
//					{
//						printf("R! l: %d, end: %d\n", c_line, end);
//						goto out_done;
//					}
//
//					for (uint32_t n = 0; n < n_words; n++)
//					{
//						buffer[n] = observations_per_class[n_i][n]
//							^ observations_per_class[n_j][n];
//
//						//lt_buffer[c_line] += __builtin_popcountl(buffer[n]);
//					}
//
//					assert(offset[0] < n_total_lines);
//
//					// Setup dataspace
//					// If writing to a portion of a dataset in a loop, be sure
//					// to close the dataspace with each iteration, as this
//					// can cause a large temporary "memory leak".
//					// "Achieving High Performance I/O with HDF5"
//					//	file_space_id = H5Dget_space(dset_id);
//					//	assert(file_space_id != NOK);
//
//					// Select hyperslab on file dataset
//					herr_t err
//						= H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET,
//											  offset, NULL, count, NULL);
//					assert(err != NOK);
//
//					// Write buffer to dataset
//					err = H5Dwrite(dset_id, H5T_NATIVE_ULONG, mem_space_id,
//								   file_space_id, xfer_plist, buffer);
//					assert(err != NOK);
//
//					// H5Sclose(file_space_id);
//
//					// Update offset
//					offset[0]+=mpi_size;
//
//					//					if (c_line > next_output)
//					//					{
//					//						fprintf(stdout,
//					//								" - Writing disjoint matrix
//					//[1/2]: %0.0f%% \r",
//					//								((double) c_line) / n_lines
//					//* 100); 						fflush(stdout);
//					//
//					//						next_output += 1; //; n_lines / 10;
//					//					}
//				}
//			}
//		}
//	}
//
// out_done:
// printf("P%d Done!\n", mpi_rank);
//
// MPI_Barrier(comm);
//	 H5Pclose(xfer_plist);
//	H5Sclose(file_space_id);
//
//	// Close memory dataspace
//	H5Sclose(mem_space_id);
//
//	free(buffer);
//
//	// Create line totals dataset
//	// write_line_totals(file_id, lt_buffer, n_lines);
//
//	free(lt_buffer);
//
//	// Close the dataset collectively
//	H5Dclose(dset_id);
//
//	return OK;
// }
//
//  oknok_t mpi_create_column_dataset(const hid_t file_id, const dataset_t*
//  dataset, 								  const int mpi_rank, const int
//  mpi_size)
//{
//	// Number of attributes
//	uint32_t n_attributes = dataset->n_attributes;
//
//	// Number of observations
//	uint32_t n_observations = dataset->n_observations;
//
//	// Number of classes
//	uint32_t n_classes = dataset->n_classes;
//
//	// Observations per class
//	word_t** observations_per_class = dataset->observations_per_class;
//
//	// Number of observations per class
//	uint32_t* n_observations_per_class = dataset->n_observations_per_class;
//
//	// Number of words in a line FROM INPUT DATASET
//	uint32_t in_n_words
//		= n_attributes / WORD_BITS + (n_attributes % WORD_BITS != 0);
//
//	// Number of lines from input dataset
//	uint32_t in_n_lines = get_dm_n_lines(dataset);
//
//	// Number of words in a line FROM OUTPUT DATASET
//	uint32_t out_n_words
//		= in_n_lines / WORD_BITS + (in_n_lines % WORD_BITS != 0);
//
//	// Round to nearest 64 so we don't have to worry when transposing
//	uint32_t out_n_lines = in_n_words * WORD_BITS;
//
//	oknok_t ret = OK;
//
//	// CREATE OUTPUT DATASET
//	// Output dataset dimensions
//	hsize_t out_dimensions[2] = { out_n_lines, out_n_words };
//
//	hid_t out_dataspace_id = H5Screate_simple(2, out_dimensions, NULL);
//	if (out_dataspace_id < 0)
//	{
//		// Error creating file
//		fprintf(stderr, "Error creating dataset space\n");
//		return NOK;
//	}
//
//	// Create a dataset creation property list
//	hid_t out_property_list_id = H5Pcreate(H5P_DATASET_CREATE);
//	// H5Pset_layout(out_property_list_id, H5D_CHUNKED);
//
//	// The choice of the chunk size affects performance!
//	// for now we will choose one line
//	// hsize_t out_chunk_dimensions[2] = { 1, out_n_words };
//
//	// H5Pset_chunk(out_property_list_id, 2, out_chunk_dimensions);
//
//	// Create the dataset
//	hid_t out_dataset_id
//		= H5Dcreate(file_id, DM_COLUMN_DATA, H5T_STD_U64LE, out_dataspace_id,
//					H5P_DEFAULT, out_property_list_id, H5P_DEFAULT);
//
//	H5Sclose(out_dataspace_id);
//
//	if (out_dataset_id < 0)
//	{
//		fprintf(stderr, "Error creating output dataset\n");
//		return NOK;
//	}
//
//	// Close resources
//	// H5Pclose(out_property_list_id);
//
//	// We're writing 64 lines at once
//	hsize_t out_mem_dimensions[2] = { WORD_BITS, out_n_words };
//
//	// Create a memory dataspace to indicate the size of our buffer/chunk
//	hid_t out_memspace_id = H5Screate_simple(2, out_mem_dimensions, NULL);
//	if (out_memspace_id < 0)
//	{
//		fprintf(stderr, "Error creating disjoint matrix memory space\n");
//		ret = NOK;
//		goto out_out_memspace;
//	}
//
//	// Allocate input buffer
//	// word_t *in_buffer = (word_t*) malloc(sizeof(word_t) * in_n_lines);
//	// Rounding to nearest multiple of 64 so we don't have to worry when
//	// transposing the last lines
//	word_t* in_buffer
//		= (word_t*) calloc(out_n_words * WORD_BITS, sizeof(word_t));
//
//	// Allocate output buffer
//	word_t* out_buffer
//		= (word_t*) calloc(out_n_words * WORD_BITS, sizeof(word_t));
//
//	// Allocate transpose buffer
//	word_t* t_buffer = calloc(WORD_BITS, sizeof(word_t));
//
//	// Allocate attribute totals buffer
//	// Correct size
//	// uint32_t *attribute_buffer = (uint32_t*) calloc(n_attributes,
//	// sizeof(uint32_t));
//	// Rounded to 64 bits
//	uint32_t* attr_buffer = (uint32_t*) calloc(out_n_lines, sizeof(uint32_t));
//
//	// Used to print out progress message
//	uint32_t next_output = 0;
//
//	word_t* current_buffer = in_buffer;
//
//	for (uint32_t i = 0; i < in_n_words; i++)
//	{
//		current_buffer = in_buffer;
//		for (uint32_t ci = 0; ci < n_classes - 1; ci++)
//		{
//			for (uint32_t cj = ci + 1; cj < n_classes; cj++)
//			{
//				for (uint32_t n_i = ci * n_observations;
//					 n_i < ci * n_observations + n_observations_per_class[ci];
//					 n_i++)
//				{
//					for (uint32_t n_j = cj * n_observations; n_j
//						 < cj * n_observations + n_observations_per_class[cj];
//						 n_j++, current_buffer++)
//					{
//						*current_buffer = observations_per_class[n_i][i]
//							^ observations_per_class[n_j][i];
//					}
//				}
//			}
//		}
//
//		// TRANSPOSE LINES
//		for (uint32_t w = 0; w < out_n_words; w++)
//		{
//			// Read 64x64 bits block from input buffer
//			//! WARNING: We may have fewer than 64 lines remaining
//			// We may be reading garbage
//			memcpy(t_buffer, in_buffer + (w * WORD_BITS),
//				   sizeof(word_t) * WORD_BITS);
//
//			// Transpose
//			transpose64(t_buffer);
//
//			// Append to output buffer
//			for (uint8_t l = 0; l < WORD_BITS; l++)
//			{
//				out_buffer[l * out_n_words + w] = t_buffer[l];
//			}
//		}
//
//		// Lets try and save 64 full lines at once!
//		hsize_t out_offset[2] = { 0, 0 };
//		hsize_t out_count[2]  = { WORD_BITS, out_n_words };
//
//		// SAVE TRANSPOSED ARRAY
//		out_offset[0] = i * WORD_BITS;
//
//		// Setup dataspace
//		// If writing to a portion of a dataset in a loop, be sure
//		// to close the dataspace with each iteration, as this
//		// can cause a large temporary "memory leak".
//		// "Achieving High Performance I/O with HDF5"
//		hid_t out_dataspace_id = H5Screate_simple(2, out_dimensions, NULL);
//
//		H5Sselect_hyperslab(out_dataspace_id, H5S_SELECT_SET, out_offset, NULL,
//							out_count, NULL);
//
//		H5Dwrite(out_dataset_id, H5T_NATIVE_ULONG, out_memspace_id,
//				 out_dataspace_id, H5P_DEFAULT, out_buffer);
//
//		//		// Lets try and save 1 full line at once!
//		//		hsize_t out_offset[2] = { 0, 0 };
//		//		hsize_t out_count[2] = { 1, out_n_words };
//		//
//		//		// SAVE TRANSPOSED ARRAY
//		//		for (uint8_t l = 0; l < WORD_BITS; l++) {
//		//			out_offset[0] = i * WORD_BITS + l;
//		//			H5Sselect_hyperslab(out_dataspace_id, H5S_SELECT_SET,
//		// out_offset, 			NULL, out_count, NULL);
//		//
//		//			H5Dwrite(out_dataset_id, H5T_NATIVE_ULONG, out_memspace_id,
//		//					out_dataspace_id, H5P_DEFAULT,
//		//					out_buffer + out_n_words * l);
//		//		}
//
//		H5Sclose(out_dataspace_id);
//
//		// Update attribute totals
//		for (uint32_t at = 0; at < WORD_BITS; at++)
//		{
//			for (uint64_t l = at * out_n_words; l < (at + 1) * out_n_words; l++)
//			{
//				attr_buffer[out_offset[0] + at]
//					+= __builtin_popcountl(out_buffer[l]);
//			}
//		}
//
//		if (i > next_output)
//		{
//			fprintf(stdout, " - Writing disjoint matrix [2/2]: %0.0f%%      \r",
//					((double) i) / in_n_words * 100);
//			fflush(stdout);
//
//			next_output += in_n_words / 10;
//		}
//	}
//
//	// Create attribute totals dataset
//	herr_t err = write_attribute_totals(file_id, attr_buffer, n_attributes);
//	if (err < 0)
//	{
//		fprintf(stderr, "Error creating attribute totals dataset\n");
//		ret = NOK;
//	}
//
//	free(t_buffer);
//	free(attr_buffer);
//	free(in_buffer);
//	free(out_buffer);
//
//  out_out_memspace:
//	H5Sclose(out_memspace_id);
//
//	// out_out_dataset:
//	H5Dclose(out_dataset_id);
//
//	return ret;
// }
//
//  herr_t mpi_write_disjoint_matrix_attributes(const hid_t dataset_id,
//											const uint32_t n_attributes,
//											const uint32_t n_matrix_lines)
//{
//	herr_t ret = 0;
//
//	ret = hdf5_write_attribute(dataset_id, N_ATTRIBUTES_ATTR, H5T_NATIVE_UINT,
//							   &n_attributes);
//	if (ret < 0)
//	{
//		return ret;
//	}
//
//	ret = hdf5_write_attribute(dataset_id, N_MATRIX_LINES_ATTR, H5T_NATIVE_UINT,
//							   &n_matrix_lines);
//
//	return ret;
// }
//
//  herr_t mpi_write_line_totals(const hid_t file_id, const uint32_t* data,
//							 const uint32_t n_lines)
//{
//	herr_t ret = 0;
//
//	// Dataset dimensions
//	hsize_t lt_dimensions[1] = { n_lines };
//
//	hid_t lt_dataset_space_id = H5Screate_simple(1, lt_dimensions, NULL);
//	if (lt_dataset_space_id < 0)
//	{
//		// Error creating file
//		fprintf(stderr, "Error creating line totals dataset space\n");
//		return lt_dataset_space_id;
//	}
//
//	// Create the dataset
//	hid_t lt_dataset_id = H5Dcreate2(file_id, DM_LINE_TOTALS, H5T_STD_U32LE,
//									 lt_dataset_space_id, H5P_DEFAULT,
//									 H5P_DEFAULT, H5P_DEFAULT);
//	if (lt_dataset_id < 0)
//	{
//		fprintf(stderr, "Error creating line totals dataset\n");
//		ret = lt_dataset_id;
//		goto out_lt_dataset_space;
//	}
//
//	herr_t err = H5Dwrite(lt_dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL,
//						  H5P_DEFAULT, data);
//	if (err < 0)
//	{
//		fprintf(stderr, "Error writing line totals\n");
//		ret = err;
//	}
//
//  out_lt_dataset_space:
//	H5Sclose(lt_dataset_space_id);
//
//	H5Dclose(lt_dataset_id);
//
//	return ret;
// }
//
//  herr_t mpi_write_attribute_totals(const hid_t file_id, const uint32_t* data,
//								  const uint32_t n_attributes)
//{
//	herr_t ret = 0;
//
//	// Dataset dimensions
//	hsize_t dimensions[1] = { n_attributes };
//
//	hid_t dataspace_id = H5Screate_simple(1, dimensions, NULL);
//	if (dataspace_id < 0)
//	{
//		// Error creating file
//		fprintf(stderr, "Error creating attribute totals dataset space\n");
//		return dataspace_id;
//	}
//
//	// Create the dataset
//	hid_t dataset_id
//		= H5Dcreate2(file_id, DM_ATTRIBUTE_TOTALS, H5T_STD_U32LE, dataspace_id,
//					 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//	if (dataset_id < 0)
//	{
//		fprintf(stderr, "Error creating attribute totals dataset\n");
//		ret = dataset_id;
//		goto out_dataspace;
//	}
//
//	herr_t err = H5Dwrite(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL,
//						  H5P_DEFAULT, data);
//	if (err < 0)
//	{
//		fprintf(stderr, "Error writing attribute totals\n");
//		ret = err;
//	}
//
//  out_dataspace:
//	H5Sclose(dataspace_id);
//
//	H5Dclose(dataset_id);
//
//	return ret;
// }
