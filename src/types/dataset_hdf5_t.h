/*
 ============================================================================
 Name        : dataset_hdf5_t.h
 Author      : Eduardo Ribeiro
 Description : Datatype representing one hgf5 dataset
 ============================================================================
 */

#ifndef HDF5_DATASET_T_H
#define HDF5_DATASET_T_H

#include "hdf5.h"

typedef struct dataset_hdf5_t
{
	/**
	 * file_id
	 */
	hid_t file_id;

	/**
	 * dataset_id
	 */
	hid_t dataset_id;

	/**
	 * Dimensions
	 */
	hsize_t dimensions[2];

} dataset_hdf5_t;

#endif // HDF5_DATASET_T_H
