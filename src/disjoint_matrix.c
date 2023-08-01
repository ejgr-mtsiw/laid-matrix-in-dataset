/*
 ============================================================================
 Name        : disjoint_matrix.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage the disjoint matrix
 ============================================================================
 */

#include "disjoint_matrix.h"

#include "dataset_hdf5.h"
#include "types/dataset_hdf5_t.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/oknok_t.h"
#include "types/word_t.h"
#include "utils/bit.h"
#include "utils/timing.h"

#include "hdf5.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

uint32_t get_dm_n_lines(const dataset_t* dataset)
{
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

oknok_t generate_dm_column(const dataset_t* dset, const int column,
						   word_t* buffer)
{
	uint32_t nc	   = dset->n_classes;
	uint32_t nobs  = dset->n_observations;
	uint32_t* nopc = dset->n_observations_per_class;
	word_t** opc   = dset->observations_per_class;

	// Current buffer line
	word_t* bl = buffer;

	/**
	 * MUST BE THE SAME ALGORITHMN / ORDER USED WHEN FILLING THE CLASS OFFSETS!
	 */
	for (uint32_t ca = 0; ca < nc - 1; ca++)
	{
		word_t** bla = opc + ca * nobs;
		for (uint32_t ia = 0; ia < nopc[ca]; ia++)
		{
			word_t* la = *(bla + ia);
			for (uint32_t cb = ca + 1; cb < nc; cb++)
			{
				word_t** blb = opc + cb * nobs;
				for (uint32_t ib = 0; ib < nopc[cb]; ib++)
				{
					// Calculate one word
					word_t* lb = *(blb + ib);

					(*bl) = la[column] ^ lb[column];
					bl++;
				}
			}
		}
	}

	return OK;
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
