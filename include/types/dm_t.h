/*
 ============================================================================
 Name        : dm_t.h
 Author      : Eduardo Ribeiro
 Description : Datatype representing one disjoint matriz
 ============================================================================
 */

#ifndef DM_T_H
#define DM_T_H

#include "types/steps_t.h"

#include <stdint.h>

typedef struct dm_t
{
	uint32_t n_matrix_lines;
	steps_t* steps;
} dm_t;

#endif // DM_T_H
