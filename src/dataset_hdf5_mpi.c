/*
 ============================================================================
 Name        : dataset_hdf5_mpi.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage HDF5 datasets
 ============================================================================
 */

#include "dataset_hdf5_mpi.h"

#include "dataset_hdf5.h"
#include "types/dataset_hdf5_t.h"
#include "types/oknok_t.h"
#include "types/word_t.h"

#include "hdf5.h"
#include "mpi.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

oknok_t mpi_hdf5_open_dataset(const char* filename, const char* datasetname,
							  const MPI_Comm comm, const MPI_Info info,
							  dataset_hdf5_t* dataset)
{
	// Setup file access template
	hid_t acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
	assert(acc_tpl != NOK);

	// Set Parallel access with communicator
	herr_t ret = H5Pset_fapl_mpio(acc_tpl, comm, info);
	assert(ret != NOK);

	// Open the file collectively
	hid_t f_id = H5Fopen(filename, H5F_ACC_RDWR, acc_tpl);
	assert(f_id != NOK);

	// Release file-access template
	ret = H5Pclose(acc_tpl);
	assert(ret != NOK);

	// Open the dataset collectively
	hid_t dset_id = H5Dopen2(f_id, datasetname, H5P_DEFAULT);
	assert(dset_id != NOK);

	dataset->file_id	= f_id;
	dataset->dataset_id = dset_id;
	hdf5_get_dataset_dimensions(dset_id, dataset->dimensions);

	return OK;
}

hid_t create_hdf5_dataset(const hid_t file_id, const char* name,
						  const uint32_t n_lines, const uint32_t n_words,
						  const hid_t datatype)
{
	// Dataset dimensions
	hsize_t dimensions[2] = { n_lines, n_words };

	hid_t filespace_id = H5Screate_simple(2, dimensions, NULL);
	assert(filespace_id != NOK);

	// Create a dataset creation property list
	hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	assert(dcpl_id != NOK);

	// Create a dataset access property list
	hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
	assert(dapl_id != NOK);

	// Create the dataset collectively
	hid_t dset_id = H5Dcreate2(file_id, name, datatype, filespace_id,
							   H5P_DEFAULT, dcpl_id, dapl_id);
	assert(dset_id != NOK);

	// Close resources
	H5Pclose(dapl_id);
	H5Pclose(dcpl_id);
	H5Sclose(filespace_id);

	return dset_id;
}

oknok_t write_n_lines(const hid_t dset_id, const uint32_t start,
					  const uint8_t n_lines_out, const uint32_t n_words,
					  const hid_t datatype, const void* buffer)
{
	/**
	 * If we don't have anything to write, return here
	 */
	if (n_lines_out == 0 || n_words == 0)
	{
		return OK;
	}

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
	err = H5Dwrite(dset_id, datatype, memspace_id, filespace_id, xfer_plist,
				   buffer);
	assert(err != NOK);

	H5Pclose(xfer_plist);
	H5Sclose(memspace_id);
	H5Sclose(filespace_id);
	return OK;
}
