/*
 ============================================================================
 Name        : utils/math.c
 Author      : Eduardo Ribeiro
 Description : Mathematical helpers
 ============================================================================
 */

#include "utils/math.h"

#include <stdint.h>

uint32_t roundUp(uint32_t numToRound, uint32_t multiple)
{
	if (multiple == 0)
	{
		return numToRound;
	}

	uint32_t remainder = numToRound % multiple;
	if (remainder == 0)
	{
		return numToRound;
	}
	return numToRound + multiple - remainder;
}
