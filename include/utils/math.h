/*
 ============================================================================
 Name        : utils/math.h
 Author      : Eduardo Ribeiro
 Description : Mathematical helpers
 ============================================================================
 */

#ifndef UTILS_MATH_H__
#define UTILS_MATH_H__

#include <stdint.h>

uint32_t roundUp(uint32_t numToRound, uint32_t multiple)
{
	if (multiple == 0)
		return numToRound;

	uint32_t remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;

	return numToRound + multiple - remainder;
}

#endif // UTILS_MATH_H__
