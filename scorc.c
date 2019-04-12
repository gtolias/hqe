#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "scorc.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* Threaded versions, if OpenMP is available */
#ifdef _OPENMP

	float * scor_compute (const int * qids, const int * dbids, const float * w, int n, int m, int burst)
	{
		int i, k;
		float wk = 0.0;
		int j = 0;
		float * scor = (float *) calloc (m, sizeof (float));

		/* use of burstiness normalization */
		if (burst)
			for (i = 0; i < n; i++)	{
				/* consecutive elements of the same query feature and database image */
				if (dbids[i] == dbids[i+1] && qids[i] == qids[i+1] && i < n-1)	{
					j++;
					wk += w[i];
				}
				/* end of consecutive elements, update scores */
				else if (j > 0)	{
					wk += w[i];
					scor[dbids[i]-1] += wk / sqrt((float)j+1);
					wk = 0.0;
					j = 0;
				}
				else
					scor[dbids[i]-1] += w[i];
			}
		/* no burstiness normalization */
		else
			for (i = 0; i < n; i++)
				scor[dbids[i]-1] += w[i];

	return scor;
	}


	int * nc_compute (const int * dbids, int n, int m)
	{
		int i, k;
		int * nc = (int *) calloc (m, sizeof (int));

		for (i = 0; i < n; i++)
			nc[dbids[i]-1] ++;

	return nc;
	}


#else  /* no OpenMP */



	float * scor_compute (const int * qids, const int * dbids, const float * w, int n, int m, int burst)
	{
		int i, k;
		float wk = 0.0;
		int j = 0;
		float * scor = (float *) calloc (m, sizeof (float));

		/* use of burstiness normalization */
		if (burst)
			for (i = 0; i < n; i++)	{
				/* consecutive elements of the same query feature and database image */
				if (dbids[i] == dbids[i+1] && qids[i] == qids[i+1] && i < n-1)	{
					j++;
					wk += w[i];
				}
				/* end of consecutive elements, update scores */
				else if (j > 0)	{
					wk += w[i];
					scor[dbids[i]-1] += wk / sqrt((float)j+1);
					wk = 0.0;
					j = 0;
				}
				else
					scor[dbids[i]-1] += w[i];
			}
		/* no burstiness normalization */
		else
			for (i = 0; i < n; i++)
				scor[dbids[i]-1] += w[i];

	return scor;
	}


	int * nc_compute (const int * dbids, int n, int m)
	{
		int i, k;
		int * nc = (int *) calloc (m, sizeof (int));

		for (i = 0; i < n; i++)
			nc[dbids[i]-1] ++;

	return nc;
	}

#endif  /*  OpenMP  */
