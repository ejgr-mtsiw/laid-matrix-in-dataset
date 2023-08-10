/*
 ============================================================================
 Name        : utils/timing.h
 Author      : Eduardo Ribeiro
 Description : Timing helpers
 ============================================================================
 */

#ifndef UTILS_TIMING_H__
#define UTILS_TIMING_H__

#include <stdio.h>
#include <time.h>

/**
 * Check if we have access to subsecond timing
 *
 * Activate with -std=gnu99 or -D_POSIX_C_SOURCE=199309L compile options
 */
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 199309L

// FreeBSD doesn't have CLOCK_MONOTONIC_RAW but has CLOCK_MONOTONIC
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

#define SETUP_TIMING                                                           \
	struct timespec tick;                                                      \
	struct timespec tock;

#define TICK                                                                   \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		clock_gettime(CLOCK_MONOTONIC_RAW, &tick);                             \
	}

#define TOCK                                                                   \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		clock_gettime(CLOCK_MONOTONIC_RAW, &tock);                             \
		fprintf(stdout, "[%0.6fs]\n",                                          \
				(tock.tv_nsec - tick.tv_nsec) / 1000000000.0F                  \
					+ (tock.tv_sec - tick.tv_sec));                            \
	}

#define SETUP_TIMING_GLOBAL                                                    \
	struct timespec global_tick;                                               \
	struct timespec global_tock;                                               \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		clock_gettime(CLOCK_MONOTONIC_RAW, &global_tick);                      \
	}

#define PRINT_TIMING_GLOBAL                                                    \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		clock_gettime(CLOCK_MONOTONIC_RAW, &global_tock);                      \
		fprintf(stdout, "[%0.6fs]\n",                                          \
				(global_tock.tv_nsec - global_tick.tv_nsec) / 1000000000.0F    \
					+ (global_tock.tv_sec - global_tick.tv_sec));              \
	}

#else
#define SETUP_TIMING                                                           \
	time_t tick = 0;                                                           \
	time_t tock = 0;

#define TICK                                                                   \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		tick = time(0);                                                        \
	}

#define TOCK                                                                   \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		tock = time(0);                                                        \
		fprintf(stdout, "[%lds]\n", tock - tick);                              \
	}

#define SETUP_TIMING_GLOBAL                                                    \
	time_t global_tick = 0;                                                    \
	time_t global_tock = 0;                                                    \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		global_tick = time(0);                                                 \
	}

#define PRINT_TIMING_GLOBAL                                                    \
	if (rank == ROOT_RANK)                                                     \
	{                                                                          \
		global_tock = time(0);                                                 \
		fprintf(stdout, "[%lds]\n", global_tock - global_tick);                \
	}

#endif // defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 199309L
#endif // UTILS_TIMING_H__
