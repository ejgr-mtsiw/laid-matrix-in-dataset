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

typedef struct steps_t
{
	uint32_t indexA;
	uint32_t indexB;
} steps_t;

#endif // STEPS_T_H
