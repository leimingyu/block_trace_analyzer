#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "cuda_intrinsics.h"
#include "cuda_warptrace.h"

#define FLTSIZE sizeof(float)

inline int BLK(int data, int blocksize)
{
	return (data + blocksize - 1) / blocksize;
}

__global__ void kernel_vectorAdd (const float* __restrict__ a_d, 
		const float* __restrict__ b_d,
		const int N,
		const int offset,
		float *c_d,
		uint *trace_sm,
		uint *trace_blk,
		double *trace_start,
		double *trace_end
		)
{
	int tid = threadIdx.x + __mul24(blockIdx.x, blockDim.x);

	// start
	trace_sm[tid] 	= get_smid();
	trace_blk[tid]	= get_global_blkid(); 
	double start_time = double(timer()) * 1e-6; 
	trace_start[tid] = start_time;

	float sum = 0.f;

	if(tid < N) {
		float a = a_d[tid + offset];
		float b = b_d[tid + offset];

		for(int i=0; i<64; i++) {
			sum += a + b;
			sum += a * b;
			sum += sum * b;
			sum += sum * a;
			sum += a / b;
			sum += log(a) + log(b);
		}

		c_d[tid + offset] = sum; 
	}

	// end
	double end_time = double(timer()) * 1e-6;
	trace_end[tid] = end_time;
}

int main( int argc, char **argv)
{
	int devid = 0 ;

	int num_streams = 8;

	int N = 1 << 20;

	int dump_trace = 0;

	if(argc >= 2)
		num_streams = atoi(argv[1]);

	if(argc >= 3)
		devid = atoi(argv[2]);

	if(argc >= 4)
		N = atoi(argv[3]);

	if(argc >= 5)
		dump_trace = atoi(argv[4]);

	cudaSetDevice(devid);

	// allocate streams
    cudaStream_t *streams = (cudaStream_t *) malloc(num_streams * sizeof(cudaStream_t));

	// init
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

	//------------------------------------------------------------------------//
	// allocate data on the host
	//------------------------------------------------------------------------//
	size_t databytes = N  * FLTSIZE; 


	float *a_h = NULL;
    checkCudaErrors(cudaMallocHost((void **)&a_h, N * num_streams * FLTSIZE));

	float *b_h = NULL;
    checkCudaErrors(cudaMallocHost((void **)&b_h, N * num_streams * FLTSIZE));

	float *c_h = NULL;
    checkCudaErrors(cudaMallocHost((void **)&c_h, N * num_streams * FLTSIZE));

	for(int i=0; i< N * num_streams; i++) {
		a_h[i] = 1.1f;	
		b_h[i] = 2.2f;	
	}

	//------------------------------------------------------------------------//
	// allocate data on the device 
	//------------------------------------------------------------------------//
	float *a_d;
	float *b_d;
	float *c_d;
	cudaMalloc((void**)&a_d, N * num_streams * FLTSIZE);
	cudaMalloc((void**)&b_d, N * num_streams * FLTSIZE);
	cudaMalloc((void**)&c_d, N * num_streams * FLTSIZE);

	// kernel configuration
	dim3 threads = dim3(256, 1, 1);
	dim3 blocks  = dim3(BLK(N, threads.x), 1, 1);

	// set up warp tracer
	WarpTrace *streams_trace = new WarpTrace[num_streams];
	for(int i=0; i<num_streams; i++) {
		streams_trace[i].compute_totalthreads(blocks, threads);
		streams_trace[i].allocate_data();
	}

	// create cuda event handles
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	cudaEventRecord(start,0);

	// copy data to deivce
	for (int i = 0; i < num_streams; i++) {
		int offset = i * N;
		cudaMemcpyAsync(&a_d[offset], &a_h[offset],  databytes, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&b_d[offset], &b_h[offset],  databytes, cudaMemcpyHostToDevice, streams[i]);
	}

	// launch one worker kernel per stream
	for (int i = 0; i < num_streams; i++) {
		int offset = i * N;
		kernel_vectorAdd <<< blocks, threads, 0, streams[i] >>> (a_d, 
				                                                 b_d, 
																 N, 
																 offset,
																 c_d,
																 streams_trace[i].trace_sm,
																 streams_trace[i].trace_blk,
																 streams_trace[i].trace_start,
																 streams_trace[i].trace_end
																 );
	}

	// copy data back to host
	for (int i = 0; i < num_streams; i++) {
		int offset = i * N;
		cudaMemcpyAsync(&c_h[offset], &c_d[offset],  databytes, cudaMemcpyDeviceToHost, streams[i]);
	}

	cudaDeviceSynchronize();
	
	float gpuTime_ms= 0;
	cudaEventElapsedTime(&gpuTime_ms, start, stop);

	//printf("runtime (ms) : %f\n", gpuTime_ms);
	if(dump_trace == 1) {

		printf("thread_id,stream_id,sm_id,block_id,start,end\n");

		for (int i = 0; i < num_streams; i++) {
			streams_trace[i].print_log(i);
		}
	}


	//------------------------------------------------------------------------//
	// free 
	//------------------------------------------------------------------------//
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

	cudaFreeHost(a_h);
	cudaFreeHost(b_h);
	cudaFreeHost(c_h);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	delete [] streams_trace;

	cudaDeviceReset();

	return 0;
}
