/*
 ============================================================================
 Name        : dm_t.h
 Author      : Eduardo Ribeiro
 Description : Datatype representing one disjoint matriz or part of one.
			   Each column corresponds to one attribute
 ============================================================================
 */

#ifndef DM_T_H
#define DM_T_H

#include "steps_t.h"

#include <stdint.h>

typedef struct dm_t
{
	/**
	 * The number of lines of the full matrix
	 */
	uint32_t n_matrix_lines;

	/**
	 * The offset in the full matrix
	 */
	uint32_t s_offset;

	/**
	 * Number of lines we can generate
	 */
	uint32_t s_size;

	/**
	 * Steps to generate the partial disjoint matrix
	 */
	steps_t* steps;
} dm_t;

#endif // DM_T_H
