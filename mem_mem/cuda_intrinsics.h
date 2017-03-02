#ifndef _CUDA_INTRINSICS_H_
#define _CUDA_INTRINSICS_H_

#include <stdint.h>

#define DEVICE_INLINE   __device__ __forceinline__                              

DEVICE_INLINE unsigned int get_smid()                                               
{                                                                               
	unsigned int r;                                                             
	asm("mov.u32 %0, %smid;" : "=r"(r));                                       
	return r;                                                                   
}                                                                               

DEVICE_INLINE unsigned int timer()                                              
{                                                                               
	unsigned int r;                                                             
	asm volatile ("mov.u32 %0, %clock;" : "=r"(r));                                      
	return r;                                                                   
}                                                                               

/*
DEVICE_INLINE unsigned long long timer64()
{                                                                               
	//unsigned int r;                                                             
	//unsigned int r_hi;                                                             

	//asm("mov.u32 %0, %clock;" : "=r"(r));                                      
	//asm("mov.u32 %0, %clock;" : "=r"(r));                                      

	//unsigned long long r;

	unsigned long long int r;


	asm("mov.u64 %0, %clock64;" : "=r"(r));                                      

	return r;                                                                   
}                                                                               
*/


DEVICE_INLINE unsigned int get_laneid(void) {                                   
	unsigned int r;                                                          
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(r));
	return r;                                                                
}                     


DEVICE_INLINE unsigned int get_warpid(void) {
	unsigned int r;
	asm volatile ("mov.u32 %0, %warpid;" : "=r"(r));
	return r;                                                                
}

DEVICE_INLINE uint4 get_ctaid(void) {                               
	uint4 r;                                                                     
	asm("mov.u32 %0, %ctaid.x;" : "=r"(r.x));                                   
	asm("mov.u32 %0, %ctaid.y;" : "=r"(r.y));                                   
	asm("mov.u32 %0, %ctaid.z;" : "=r"(r.z));                                   
	return r;                                                                   
} 

// https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
DEVICE_INLINE unsigned int get_global_blkid(){
	return blockIdx.x + gridDim.x * blockIdx.y 
		+ gridDim.x * gridDim.y * blockIdx.z; 
}


DEVICE_INLINE unsigned int get_global_warpid(){
	int warps_blk_x = int(ceil(blockDim.x / 32.f));
	int warps_blk_y = int(ceil(blockDim.y / 32.f));

	int warps_Dimx = gridDim.x * warps_blk_x; 
	int warps_Dimy = gridDim.y * warps_blk_y; 

	//int warps_Dimx = int(ceil(gridDim.x / 32.f));
	//int warps_Dimy = int(ceil(gridDim.y / 32.f));

	//int global_warpid = get_warpid() + blockIdx.x * warps_blk_x;

	int global_warpid = get_warpid() + blockIdx.x * warps_blk_x + 
		warps_Dimx * blockIdx.y + 
		warps_Dimx * warps_Dimy * blockIdx.z;

	return global_warpid;
}

#endif
