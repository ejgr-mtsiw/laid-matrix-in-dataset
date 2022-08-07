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

oknok_t generate_dm_column(const dataset_t* dset, const dm_t* dm,
						   const int column, word_t* buffer)
{
	word_t* current_buffer = buffer;

	steps_t* s	   = dm->steps;
	steps_t* s_end = s + dm->n_matrix_lines;

	for (; s < s_end; s++)
	{
		word_t* la = dset->data + (s->indexA * dset->n_words) + column;
		word_t* lb = dset->data + (s->indexB * dset->n_words) + column;

		*current_buffer = *la ^ *lb;
		current_buffer++;
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

void free_dm(dm_t* dm)
{
	dm->n_matrix_lines = 0;
	dm->s_offset	   = 0;
	dm->s_size		   = 0;
	free(dm->steps);
	dm->steps = NULL;
}
