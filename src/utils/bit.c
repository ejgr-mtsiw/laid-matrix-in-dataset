/*
 ============================================================================
 Name        : utils/bit.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manipulate bits
 ============================================================================
 */

#include "utils/bit.h"
#include "types/word_t.h"

#include <stdint.h>

word_t set_bits(const word_t destination, const word_t source, const uint8_t at,
				const uint8_t numbits)
{
	word_t mask = (((word_t) (~0LU)) >> (WORD_BITS - numbits)) << at;
	return (destination & ~mask) | ((source << at) & mask);
}

word_t invert_n_bits(word_t source, uint8_t numbits)
{
	if (source == 0)
	{
		return source;
	}

	word_t r_source = source;
	r_source >>= numbits;

	while (numbits--)
	{
		r_source <<= 1;
		r_source |= (source & 1UL);
		source >>= 1;
	}

	return r_source;
}

word_t get_bits(const word_t source, const uint8_t at, const uint8_t numbits)
{
	word_t mask = ((word_t) (~0LU)) << numbits;
	return (source >> at) & ~mask;
}

/**
 * Based on Hacker's Delight book
 * https://stackoverflow.com/questions/41778362/
 * how-to-efficiently-transpose-a-2d-bit-matrix
 */
void transpose64(uint64_t a[64])
{
	int j, k;
	uint64_t m, t;

	for (j = 32, m = 0x00000000FFFFFFFF; j; j >>= 1, m ^= m << j)
	{
		for (k = 0; k < 64; k = ((k | j) + 1) & ~j)
		{
			t = (a[k] ^ (a[k | j] >> j)) & m;
			a[k] ^= t;
			a[k | j] ^= (t << j);
		}
	}
}
