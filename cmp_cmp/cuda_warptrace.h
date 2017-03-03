#ifndef _CUDA_WARPTRACE_H_
#define _CUDA_WARPTRACE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>


class WarpTrace
{
public:
	WarpTrace() {
		trace_sm 		= NULL;
		trace_blk 		= NULL;
		trace_start   	= NULL;
		trace_end     	= NULL;
		total_warps   	= 0;
		total_threads 	= 0;
	}

	~WarpTrace() {
		free_data();	
	}

	void compute_totalwarps(dim3 blocks, dim3 threads)
	{
		int total_blocks = blocks.x * blocks.y * blocks.z;	

		int warps_per_block = int(ceil(threads.x / 32.f) * 
				ceil(threads.y / 32.f) * ceil(threads.z / 32.f));

		total_warps = total_blocks * warps_per_block;
	}

	void compute_totalthreads(dim3 blocks, dim3 threads)
	{
		int total_blocks = blocks.x * blocks.y * blocks.z;	
		int threads_per_blk = threads.x * threads.y * threads.z; 
		total_threads= total_blocks * threads_per_blk;
	}

	void print_log(int streamid)
	{
		cudaDeviceSynchronize();

		for(int i=0; i<total_threads; i++)
		{
			printf("%12d,%12d,%12u,%12u,%12lf,%12lf\n",
					i, 
					streamid, 
					trace_sm[i], 
					trace_blk[i],
					trace_start[i],
					trace_end[i]);
		}
	}

	void allocate_data() 
	{
		cudaMallocManaged((void**)&trace_sm,    total_threads * sizeof(uint));
		cudaMallocManaged((void**)&trace_blk,   total_threads * sizeof(uint));
		cudaMallocManaged((void**)&trace_start, total_threads * sizeof(double));
		cudaMallocManaged((void**)&trace_end,   total_threads * sizeof(double));
	}

	void free_data()
	{
		if(trace_sm    != NULL)	cudaFree(trace_sm);
		if(trace_blk   != NULL)	cudaFree(trace_blk);
		if(trace_start != NULL)	cudaFree(trace_start);
		if(trace_end   != NULL)	cudaFree(trace_end);
	}

	uint* trace_sm;
	uint* trace_blk;
	double* trace_start;
	double* trace_end;

	uint total_warps;
	uint total_threads;
};


#endif
