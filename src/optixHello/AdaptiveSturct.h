#pragma once
#include <optixu/optixu_math_namespace.h>
#include <optix_device.h>

//#ifdef __CUDACC__
//#define CUDA_CALLABLE_MEMBER __host__ __device__
//#else
//#define CUDA_CALLABLE_MEMBER
//#endif 

using namespace optix;

struct AdaptiveStruct 
{
public:
	buffer<int4, 2>* _adaptive_samples_budget_buffer;

	//unsigned int _max_total_samples_budget;

	//unsigned int _max_per_frame_samples_budget;

	//uint _window_size;

	//bool _initialized;
};