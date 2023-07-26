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
