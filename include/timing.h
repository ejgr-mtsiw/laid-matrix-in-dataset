/*
 ============================================================================
 Name        : globals.h
 Author      : Eduardo Ribeiro
 Description : Global variables definitions
 ============================================================================
 */

#ifndef TIMING_H__
#define TIMING_H__

#include <time.h>

// FreeBSD doesn't have CLOCK_MONOTONIC_RAW
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

#define SETUP_TIMING struct timespec tick, tock;
#define TICK		 clock_gettime(CLOCK_MONOTONIC_RAW, &tick);
#define TOCK(stream)                                                           \
	clock_gettime(CLOCK_MONOTONIC_RAW, &tock);                                 \
	fprintf(stream, "[%0.3fs]\n",                                              \
			(tock.tv_nsec - tick.tv_nsec) / 1000000000.0F                      \
				+ (tock.tv_sec - tick.tv_sec));

#endif // TIMING_H__
