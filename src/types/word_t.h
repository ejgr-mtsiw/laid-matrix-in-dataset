
/*
 ============================================================================
 Name        : word.h
 Author      : Eduardo Ribeiro
 Description : Defines the word_t datatype that we'll use to manipulate bits
 ============================================================================
 */

#ifndef WORD_T_H__
#define WORD_T_H__

#include <limits.h>
#include <stdint.h>

// One word is the basic block of data
typedef uint64_t word_t;

// Size of our words in bits
#define WORD_BITS (CHAR_BIT * sizeof(word_t))

#endif // WORD_T_H__
