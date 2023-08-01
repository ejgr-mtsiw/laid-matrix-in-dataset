/*
 * class_offsets_t.h
 *
 * Created on: 29/07/2023
 * Author: Eduardo Ribeiro
 * Description: Datatype representing the initial classes
 * and offsets for a process to generate the disjoint matrix
 */

#ifndef TYPES_CLASS_OFFSETS_T_H_
#define TYPES_CLASS_OFFSETS_T_H_

#include <stdint.h>

/**
 * Classes and corresponding offsets for the first step in the
 * disjoint matrix
 */
typedef struct class_offsets_t
{
	uint32_t classA;
	uint32_t indexA;
	uint32_t classB;
	uint32_t indexB;
} class_offsets_t;

#endif /* TYPES_CLASS_OFFSETS_T_H_ */
