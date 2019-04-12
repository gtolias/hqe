/* Majority voting on binary signatures */

#ifndef __scor_h__
#define __scor_h__

#include <stdio.h>


/* compute score per image */
float * scor_compute (const int * qids, const int * dbids, const float * w, int n, int m, int burst);

/* compute the number of matches per image */
int * nc_compute (const int * dbids, int n, int m);

#endif
