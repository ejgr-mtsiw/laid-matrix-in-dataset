/*
 ============================================================================
 Name        : jnsq.h
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage JNSQ
 ============================================================================
 */

#ifndef JNSQ_H
#define JNSQ_H

/**
 * Replaces the class bits with jnsq bits
 * It's ok, because the number of jnsq bits is always <= bits needed for class
 * And we don't need the class anymore because we extracted it to the
 * dataset_line structure.
 * And we fill the bits in reverse jnsq=1 with 3bits = 1 0 0
 * So the extra bits are zeroed and can be ignored in the calculations
 *
 * Also inconsistency = class
 * See Apol?nia, J., & Cavique, L. (2019). Sele??o de Atributos de Dados
 * Inconsistentes em ambiente HDF5+ Python na cloud INCD. Revista de
 * Ci?ncias da Computa??o, 85-112.
 */
void set_jnsq_bits(word_t* line, uint32_t inconsistency,
				   const uint32_t n_attributes, const uint32_t n_words,
				   const uint8_t n_bits_for_class);

/**
 * Compares 2 lines and updates jnsq on to_update if needed and updates
 * inconsistency level
 */
void update_jnsq(word_t* to_update, const word_t* to_compare,
				 uint32_t* inconsistency, const uint32_t n_attributes,
				 const uint32_t n_words, const uint8_t n_bits_for_class);

/**
 * Adds the JNSQs attributes to the dataset.
 * Returns max inconsistency found
 */
uint32_t add_jnsqs(dataset_t* dataset);

#endif
