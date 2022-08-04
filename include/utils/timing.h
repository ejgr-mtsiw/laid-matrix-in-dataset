/*
 ============================================================================
 Name        : utils/timing.h
 Author      : Eduardo Ribeiro
 Description : Timing helpers
 ============================================================================
 */

#ifndef UTILS_TIMING_H__
#define UTILS_TIMING_H__

#include <time.h>

// FreeBSD doesn't have CLOCK_MONOTONIC_RAW
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

#define SETUP_TIMING struct timespec tick, tock;

#define TICK clock_gettime(CLOCK_MONOTONIC_RAW, &tick);

#define TOCK(stream)                                                           \
	clock_gettime(CLOCK_MONOTONIC_RAW, &tock);                                 \
	fprintf(stream, "[%0.3fs]\n",                                              \
			(tock.tv_nsec - tick.tv_nsec) / 1000000000.0F                      \
				+ (tock.tv_sec - tick.tv_sec));

//#define SETUP_TIMING
//#define TICK
//#define TOCK(stream) fprintf(stream, "\n");

#endif // UTIL_TIMING_H__
