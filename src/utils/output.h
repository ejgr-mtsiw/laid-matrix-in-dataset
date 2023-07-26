/*
 ============================================================================
 Name        : utils/output.h
 Author      : Eduardo Ribeiro
 Description : Output helpers
 ============================================================================
 */

#ifndef UTILS_OUTPUT_H__
#define UTILS_OUTPUT_H__

#include "utils/ranks.h"
#include <stdio.h>

#define SAY(what)                                                              \
	fprintf(stdout, "[%d] ", rank);                                            \
	fprintf(stdout, what);

#define ROOT_SAYS(what)                                                        \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		fprintf(stdout, what);                                                 \
	}

#define LOCAL_ROOT_SAYS(what)                                                  \
	if (node_rank == LOCAL_ROOT_RANK)                                          \
	{                                                                          \
		fprintf(stdout, what);                                                 \
	}

#define SHOW(format, what)                                                     \
	fprintf(stdout, "[%d] ", rank);                                            \
	fprintf(stdout, format, what);

#define ROOT_SHOWS(format, what)                                               \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		fprintf(stdout, format, what);                                         \
	}

#define LOCAL_ROOT_SHOWS(format, what)                                         \
	if (node_rank == LOCAL_ROOT_RANK)                                          \
	{                                                                          \
		fprintf(stdout, format, what);                                         \
	}

#endif // UTILS_OUTPUT_H__
