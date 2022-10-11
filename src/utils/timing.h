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

#define SETUP_TIMING time_t tick = 0, tock = 0;

#define TICK tick = time(0); // clock_gettime(CLOCK_MONOTONIC_RAW, &tick);

#define TOCK(stream) tock=time(0); fprintf(stream, "[%lds]\n", tock - tick);

#endif // UTIL_TIMING_H__
