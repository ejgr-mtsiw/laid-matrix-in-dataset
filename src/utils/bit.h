/*
 ============================================================================
 Name        : utils/bit.h
 Author      : Eduardo Ribeiro
 Description : Utils for bit manipulation
 ============================================================================
 */

#ifndef UTILS_BIT_H
#define UTILS_BIT_H

#include "types/word_t.h"

#include <stdint.h>

static const word_t AND_MASK_TABLE[64] = { 0x1,
										   0x2,
										   0x4,
										   0x8,
										   0x10,
										   0x20,
										   0x40,
										   0x80,
										   0x100,
										   0x200,
										   0x400,
										   0x800,
										   0x1000,
										   0x2000,
										   0x4000,
										   0x8000,
										   0x10000,
										   0x20000,
										   0x40000,
										   0x80000,
										   0x100000,
										   0x200000,
										   0x400000,
										   0x800000,
										   0x1000000,
										   0x2000000,
										   0x4000000,
										   0x8000000,
										   0x10000000,
										   0x20000000,
										   0x40000000,
										   0x80000000,
										   0x100000000,
										   0x200000000,
										   0x400000000,
										   0x800000000,
										   0x1000000000,
										   0x2000000000,
										   0x4000000000,
										   0x8000000000,
										   0x10000000000,
										   0x20000000000,
										   0x40000000000,
										   0x80000000000,
										   0x100000000000,
										   0x200000000000,
										   0x400000000000,
										   0x800000000000,
										   0x1000000000000,
										   0x2000000000000,
										   0x4000000000000,
										   0x8000000000000,
										   0x10000000000000,
										   0x20000000000000,
										   0x40000000000000,
										   0x80000000000000,
										   0x100000000000000,
										   0x200000000000000,
										   0x400000000000000,
										   0x800000000000000,
										   0x1000000000000000,
										   0x2000000000000000,
										   0x4000000000000000,
										   0x8000000000000000 };

/* a=target variable, b=bit number to act upon 0-n */
#define BIT_SET(a, b)			   ((a) |= ((word_t) 1UL << (b)))
#define BIT_CLEAR(a, b)			   ((a) &= ~((word_t) 1UL << (b)))
#define BIT_FLIP(a, b)			   ((a) ^= ((word_t) 1UL << (b)))
#define BIT_CHECK(a, b)			   (!!((a) & ((word_t) 1UL << (b))))
#define BITMASK_SET(x, mask)	   ((x) |= (mask))
#define BITMASK_CLEAR(x, mask)	   ((x) &= (~(mask)))
#define BITMASK_FLIP(x, mask)	   ((x) ^= (mask))
#define BITMASK_CHECK_ALL(x, mask) (!(~(x) & (mask)))
#define BITMASK_CHECK_ANY(x, mask) ((x) & (mask))

/**
 * An interesting problem I've been pondering the past few days is how to copy
 * one integer's bits into another integer at a given position in the
 * destination integer. So, for example, given the destination integer
 * 0xdeadbeef and the source integer 0xabcd, the idea would be to get a result
 * of 0xabcdbeef (given a destination position of 16 bits) or 0xdeabcdef (given
 * a destination position of 8 bits). With the arbitrary limitation of avoiding
 * conditionals or loops (allowing myself to use just mathematical/bitwise
 * operations), I developed the following function (C++)
 * https://stackoverflow.com/questions/1283221/
 * algorithm-for-copying-n-bits-at-arbitrary-position-from-one-int-to-another
 */
word_t set_bits(const word_t destination, const word_t source, const uint8_t at,
				const uint8_t numbits);

/**
 * Inverts the last numbits of source
 * Ex:
 * source: 00000101
 * numbits: 4
 * inverted: 00001010
 */
word_t invert_n_bits(word_t source, uint8_t numbits);

/**
 * Return numbits from source starting at at
 */
word_t get_bits(const word_t source, const uint8_t at, const uint8_t numbits);

/**
 * Transposes a 64x64 bit matrix
 */
void transpose64(uint64_t a[64]);

#endif // UTILS_BIT_H
