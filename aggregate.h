/* Majority voting on binary signatures */

#ifndef __aggregate_h
#define __aggregate_h

#include <stdio.h>


typedef unsigned char uint8;

float * aggregate_descr (const float * bs, const int * keys, int m, int n, int nt);
float * aggregate_sim (const int * dbids, const float * w, int n, int m);
/* aggregate bit vectors in compact mode (bs) per key value (keys). output is not in compact mode */
float * aggregate_binary (const uint8 * bs, const int * keys, int m, int n, int nt);

#endif
