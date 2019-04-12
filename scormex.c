#include "scorc.h"
#include "mex.h"
#include <string.h>

typedef unsigned char uint8;


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  int i;

	if (nlhs != 1 || (nrhs != 5 && nrhs != 2 ))
		mexErrMsgTxt ("Usage: sc = scoremex (query_ids, db_ids, similarities, db_size, burst_flat) or nc = scoremex (dbids, db_size)");

	if (nrhs == 5)	{
		int n = mxGetM (prhs[0]);
		int * qids = (int *) mxGetPr (prhs[0]);	
		int * dbids = (int *) mxGetPr (prhs[1]);
		float * w = (float *) mxGetPr (prhs[2]);
		int m = (int) mxGetScalar (prhs[3]);
		int burst = (int) mxGetScalar (prhs[4]);

	 	float * sc_ = scor_compute (qids, dbids, w, n, m, burst);  
	
		plhs[0] = mxCreateNumericMatrix (1, m, mxSINGLE_CLASS, mxREAL);
		float * sc = (float *) mxGetPr (plhs[0]);
		memcpy (sc, sc_, sizeof(float) * m);

		free (sc_);
	}
	else	{
		int n = mxGetM (prhs[0]);
		int * dbids = (int *) mxGetPr (prhs[0]);
		int m = (int) mxGetScalar (prhs[1]);
		
	 	int * nc_ = nc_compute (dbids, n, m);  
	
		plhs[0] = mxCreateNumericMatrix (1, m, mxINT32_CLASS, mxREAL);
		int * nc = (int *) mxGetPr (plhs[0]);
		memcpy (nc, nc_, sizeof(int) * m);

		free (nc_);
	}
}
