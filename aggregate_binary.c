#include "aggregate.h"
#include "mex.h"
#include <string.h>

void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  int i;

	if (nlhs != 1 || nrhs != 3 )
		mexErrMsgTxt ("Usage: agr_bs = aggregate_binary (bs, keys, max_key)");

  int m = mxGetM (prhs[0]);
  int n = mxGetN (prhs[0]);
  uint8 * bs = (uint8 *) mxGetPr (prhs[0]);
  int * keys = (int *) mxGetPr (prhs[1]);
  int nt = (int) mxGetScalar (prhs[2]);

	/* keys should start from 0*/
  for (i = 0; i < n ; i++)
		if (keys[i] > nt)	
			mexErrMsgTxt ("key larger than maximum key value provided");
		else
			keys[i]--;
	

	float * mvec_ = aggregate_binary(bs, keys, m, n, nt);

	plhs[0] = mxCreateNumericMatrix (8 * m, nt, mxSINGLE_CLASS, mxREAL);
	float * mvec = (float *) mxGetPr (plhs[0]);
	memcpy (mvec, mvec_, sizeof(float) * 8 * m * nt);

	free(mvec_);
}
