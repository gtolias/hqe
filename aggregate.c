#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "aggregate.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* Threaded versions, if OpenMP is available */
#ifdef _OPENMP

float * aggregate_descr (const float * bs, const int * keys, int m, int n, int nt)
{
	int i, j, k;
	uint8 v;

/*	float * mvec = (float *) calloc (8 * m * nt, sizeof (float)); */
	float * mvec = (float *) calloc (m * nt, sizeof (float));
	unsigned int * h = (unsigned int *) calloc (nt, sizeof (unsigned int));
	
	/* count number of vectors for each key */
	for (i = 0; i < n; i++)
		h[keys[i]]++;

	/* sum ... */
  for (i = 0; i < n ; i++)	{
		for (j = 0; j < m ; j++)	{

			mvec[keys[i] * m + j] += bs[i * m +j];

		}
	}
	
	/* ... and average */
  for (i = 0; i < m ; i++)	{
		for (j = 0; j < nt ; j++)	{
			mvec[j * m + i] /= h[j];
		}
	}

	free(h);

return mvec;
}

float * aggregate_sim (const int * dbids, const float * w, int n, int m)
{
	float * scor = (float *) calloc (m, sizeof (float));
	int i;
	for (i = 0; i < n; i++)
		scor[dbids[i]-1] += w[i];

return scor;
}

float * aggregate_binary (const uint8 * bs, const int * keys, int m, int n, int nt)
{
	int i, j, k;

	float * mvec = (float *) calloc (8 * m * nt, sizeof (float));
	unsigned int * h = (unsigned int *) calloc (nt, sizeof (unsigned int));

	/* count number of vectors for each key */
	for (i = 0; i < n; i++)
		h[keys[i]]++;

#pragma omp parallel shared (mvec, keys, m, n) private (i, j, k)
{
#pragma omp for 
	/* sum ... */
  for (i = 0; i < n ; i++)	{
		for (j = 0; j < m ; j++)	{

			uint8 v = bs[i * m + j];

			for (k = 0; k < 8 ; k++)	{

				mvec[keys[i] * 8 * m + j * 8 + k] += (v & 1);
				v = v >> 1;

			}
		}
	}
}

#pragma omp parallel shared (mvec, m, nt, h) private (i, j)
{
#pragma omp for 
	/* ... and average */
  for (i = 0; i < 8 * m ; i++)	{
		for (j = 0; j < nt ; j++)	{
			mvec[j * 8 * m + i] /= h[j];
		}
	}
}

	free(h);

return mvec;
}


#else  /* no _OPENMP */

float * aggregate_descr (const float * bs, const int * keys, int m, int n, int nt)
{
	int i, j, k;
	uint8 v;

/*	float * mvec = (float *) calloc (8 * m * nt, sizeof (float)); */
	float * mvec = (float *) calloc (m * nt, sizeof (float));
	unsigned int * h = (unsigned int *) calloc (nt, sizeof (unsigned int));
	
	/* count number of vectors for each key */
	for (i = 0; i < n; i++)
		h[keys[i]]++;

	/* sum ... */
  for (i = 0; i < n ; i++)	{
		for (j = 0; j < m ; j++)	{

			mvec[keys[i] * m + j] += bs[i * m +j];

		}
	}
	
	/* ... and average */
  for (i = 0; i < m ; i++)	{
		for (j = 0; j < nt ; j++)	{
			mvec[j * m + i] /= h[j];
		}
	}

	free(h);

return mvec;
}

float * aggregate_sim (const int * dbids, const float * w, int n, int m)
{
	float * scor = (float *) calloc (m, sizeof (float));
	int i;
	for (i = 0; i < n; i++)
		scor[dbids[i]-1] += w[i];

return scor;
}

float * aggregate_binary (const uint8 * bs, const int * keys, int m, int n, int nt)
{
	int i, j, k;
	uint8 v;

	float * mvec = (float *) calloc (8 * m * nt, sizeof (float));
	unsigned int * h = (unsigned int *) calloc (nt, sizeof (unsigned int));
	
	/* count number of vectors for each key */
	for (i = 0; i < n; i++)
		h[keys[i]]++;

	/* sum ... */
  for (i = 0; i < n ; i++)	{
		for (j = 0; j < m ; j++)	{

			v = bs[i * m + j];

			for (k = 0; k < 8 ; k++)	{

				mvec[keys[i] * 8 * m + j * 8 + k] += (v & 1);
				v = v >> 1;

			}
		}
	}
	
	/* ... and average */
  for (i = 0; i < 8 * m ; i++)	{
		for (j = 0; j < nt ; j++)	{
			mvec[j * 8 * m + i] /= h[j];
		}
	}

	free(h);

return mvec;
}

#endif  /*_OPENMP*/

