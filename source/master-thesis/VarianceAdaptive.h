#include <optixu/optixu_math_namespace.h>
#include <optix_device.h>

//
// A way to implement and use classes in cuda code
//
//#ifdef __CUDACC__
//#define CUDA_CALLABLE_MEMBER __host__ __device__
//#else
//#define CUDA_CALLABLE_MEMBER
//#endif 

using namespace optix;

rtDeclareVariable(unsigned int, window_size, , );
rtDeclareVariable(unsigned int, max_per_frame_samples_budget, , ) = static_cast<uint>(5u);		/* this variable can be written by the user */
rtDeclareVariable(int, camera_changed, , );
rtBuffer<int4, 2>	  adaptive_samples_budget_buffer;									/* this buffer will be initialized by the host, but must also be modified by the graphics device */


rtBuffer<float4, 2>	  per_window_variance_buffer_output;

rtBuffer<float4, 2>   input_buffer;	

struct VarianceAdaptiveStruct
{
	buffer<int4, 2>* _adaptive_samples_budget_buffer;
	buffer<float4, 2>* _input_buffer; 
	buffer<float4, 2>* _per_window_variance_buffer_output;
	
	unsigned int _max_per_frame_samples_budget;
	uint _window_size;
};

static __device__ float compute_window_variance(uint2 center, uint window_size, 
															buffer<float4, 2>* input_buffer, buffer<float4, 2>* per_window_variance_buffer_output)
{
	uint2 screen = make_uint2(input_buffer->size().x, input_buffer->size().y);

	float mean = 0.f;
	float variance = 0.f;
	if ((*per_window_variance_buffer_output)[center].x < 0.0f)
	{
		uint squared_window_size = window_size * window_size;
		uint half_window_size = (window_size / 2) + (window_size % 2);
		uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

		/* compute mean value */
		for (uint i = 0; i < squared_window_size; i++)
		{
			uint2 idx = make_uint2((i % window_size + top_left_window_corner.x) % screen.x, (i / window_size + top_left_window_corner.y) % screen.y);
			float3 input_buffer_val = make_float3((*input_buffer)[idx].x, (*input_buffer)[idx].y, (*input_buffer)[idx].z);
			mean += 1.f / 3.f * (input_buffer_val.x + input_buffer_val.y + input_buffer_val.z);
		}

		/*mean *= 1.f/ squared_window_size;*/
		mean = 1.f / squared_window_size * mean;

		/* compute variance */
		for (uint i = 0; i < squared_window_size; i++)
		{
			uint2 idx = make_uint2((i % window_size + top_left_window_corner.x) % screen.x, (i / window_size + top_left_window_corner.y) % screen.x);
			float3 input_buffer_val = make_float3((*input_buffer)[idx].x, (*input_buffer)[idx].y, (*input_buffer)[idx].z);
			float var = 1.f / 3.f * (input_buffer_val.x + input_buffer_val.y + input_buffer_val.z);
			/*variance += var * var;*/
			variance += (var * var - 2.0f * mean * var + mean * mean);
		}

		//variance = 1.f / squared_window_size * (variance) - (mean * mean);
		variance = 1.f / squared_window_size * variance;

		//(*per_window_variance_buffer_output)[center] = make_float4(variance);
		atomicExch(&per_window_variance_buffer_output[center].x, variance);
	}
	else
	{
		//rtPrintf("Reuse variance!!!\n\n");
		variance = (*per_window_variance_buffer_output)[center].x;
	}

	return variance;
};

static __device__ uint compute_samples_number(uint2 current_launch_index, float variance, 
														buffer<int4, 2>* adaptive_samples_budget_buffer, unsigned int max_per_frame_samples_budget)
{
	uint samples_number = 0;

	if ((*adaptive_samples_budget_buffer)[current_launch_index].x > 0)
	{
		samples_number = static_cast<uint>(clamp(static_cast<float>(variance * max_per_frame_samples_budget), 0.0f, static_cast<float>(max_per_frame_samples_budget)));
		(*adaptive_samples_budget_buffer)[current_launch_index] = make_int4((*adaptive_samples_budget_buffer)[current_launch_index].x - static_cast<int>(samples_number));
	}

	return samples_number;
};

static __device__ uint compute_current_samples_number(uint2 current_launch_index, VarianceAdaptiveStruct variance_adaptive_struct)
{
	uint sample_number = 0;

	size_t2 screen = variance_adaptive_struct._input_buffer->size();

	uint times_width = screen.x / variance_adaptive_struct._window_size;
	uint times_height = screen.y / variance_adaptive_struct._window_size;

	uint horizontal_padding = static_cast<uint>(0.5f * (screen.x - (times_width * variance_adaptive_struct._window_size)));
	uint vertical_padding = static_cast<uint>(0.5f * (screen.y - (times_height * variance_adaptive_struct._window_size)));

	uint half_window_size = (variance_adaptive_struct._window_size / 2) + (variance_adaptive_struct._window_size % 2);

	uint2 times_launch_index = make_uint2(((current_launch_index.x / variance_adaptive_struct._window_size) * variance_adaptive_struct._window_size) % screen.x, ((current_launch_index.y / variance_adaptive_struct._window_size) * variance_adaptive_struct._window_size) % screen.y);

	uint2 current_window_center = make_uint2(times_launch_index.x + horizontal_padding + half_window_size, times_launch_index.y + vertical_padding + half_window_size);

	float variance = compute_window_variance(current_window_center, variance_adaptive_struct._window_size, variance_adaptive_struct._input_buffer, variance_adaptive_struct._per_window_variance_buffer_output);

	sample_number = compute_samples_number(current_launch_index, (30.0f * variance), variance_adaptive_struct._adaptive_samples_budget_buffer, variance_adaptive_struct._max_per_frame_samples_budget);

	return sample_number;
};