/*
 ============================================================================
 Name        : steps_t.h
 Author      : Eduardo Ribeiro
 Description : Datatype representing a disjoint matrix step
 ============================================================================
 */

#ifndef STEPS_T_H
#define STEPS_T_H

#include <stdint.h>

/**
 * Indexes on the original dataset needed to assemble
 * one line of the disjoint matrix
 */
typedef struct steps_t
{
	uint32_t indexA;
	uint32_t indexB;
} steps_t;

#endif // STEPS_T_H
