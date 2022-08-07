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
	/* setup file access template */
	hid_t acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
	assert(acc_tpl != NOK);
	/* set Parallel access with communicator */
	herr_t ret = H5Pset_fapl_mpio(acc_tpl, comm, info);
	assert(ret != NOK);

	/* open the file collectively */
	hid_t fid = H5Fopen(filename, H5F_ACC_RDWR, acc_tpl);
	assert(fid != NOK);

	/* Release file-access template */
	ret = H5Pclose(acc_tpl);
	assert(ret != NOK);

	/* open the dataset collectively */
	hid_t dsetid = H5Dopen2(fid, datasetname, H5P_DEFAULT);
	assert(dsetid != NOK);

	dataset->file_id	= fid;
	dataset->dataset_id = dsetid;
	hdf5_get_dataset_dimensions(dsetid, dataset->dimensions);

	return OK;
}
