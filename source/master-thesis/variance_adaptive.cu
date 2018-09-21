//#include <optixu/optixu_math_namespace.h>
////#include "optixPathTracer.h"
//
//using namespace optix;
//
//// Adaptive post processing variables and buffers
//
//rtDeclareVariable(unsigned int, window_size, , );
////rtDeclareVariable(unsigned int, max_ray_budget_total, , ) = static_cast<uint>(50u);
//rtDeclareVariable(unsigned int, max_per_frame_samples_budget, , ) = static_cast<uint>(5u);		/* this variable can be written by the user */
//
////
//// Adaptive version of pathtracing begin
////
//
///*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
///* Adaptive additional rays variables */
//rtBuffer<int4, 2>	  adaptive_samples_budget_buffer;									/* this buffer will be initialized by the host, but must also be modified by the graphics device */
//																						//rtBuffer<int4, 2>	  hoelder_refinement_buffer;										/* this buffer contains the information, where refinement will take place according to
//																						//hoelder regularity criterion, everywhere where refinement is needed value is 1, else zero */
//
//// For debug!
//rtBuffer<float4, 2>	  per_window_variance_buffer_output;
//
//rtBuffer<float4, 2>   input_buffer;														/* this buffer contains the initially rendered picture to be post processed */
//
////rtBuffer<float4, 2>   post_process_output_buffer;										/* this buffer contains the result, processed with additional adaptive rays */
//
//static __device__ __inline__ float compute_window_variance(uint2 center, uint window_size)
//{
//	uint2 screen = make_uint2(input_buffer.size().x, input_buffer.size().y);
//
//	float mean = 0.f;
//	float variance = 0.f;
//	if (per_window_variance_buffer_output[center].x < 0.0f)
//	{
//		uint squared_window_size = window_size * window_size;
//		uint half_window_size = (window_size / 2) + (window_size % 2);
//		uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);
//
//		//rtPrintf("Top left window corner: [ %d , %d ]\n\n", top_left_window_corner.x, top_left_window_corner.y);
//		//post_process_output_buffer[center] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
//
//		/* compute mean value */
//		for (uint i = 0; i < squared_window_size; i++)
//		{
//			uint2 idx = make_uint2((i % window_size + top_left_window_corner.x) % screen.x, (i / window_size + top_left_window_corner.y) % screen.y);
//			float3 input_buffer_val = make_float3(input_buffer[idx].x, input_buffer[idx].y, input_buffer[idx].z);
//			mean += 1.f / 3.f * (input_buffer_val.x + input_buffer_val.y + input_buffer_val.z);
//			//if (i % window_size <= i / window_size)
//			//{
//			//	post_process_output_buffer[idx] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
//			//}
//			//if (i % window_size == 0 && i / window_size == 0)
//			//{
//			//	//rtPrintf("Left lower corner, with window size: %d!!! \n\n", window_size);
//			//	//rtPrintf("Center: [ %d , %d ], Current global window index: [ %d , %d ] \n\n", center.x, center.y, idx.x, idx.y);
//			//	post_process_output_buffer[idx] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
//			//}
//			//if (i % window_size == window_size - 1 && i / window_size == 0)
//			//{
//			//	post_process_output_buffer[idx] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
//			//}
//			//if (i % window_size == 0 && i / window_size == window_size - 1)
//			//{
//			//	post_process_output_buffer[idx] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
//			//}
//			//if (i % window_size == window_size - 1 && i / window_size == window_size - 1)
//			//{
//			//post_process_output_buffer[idx] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
//			//post_process_output_buffer[top_left_window_corner] = make_float4(100.0f, 0.0f, 0.0f, 1.0f);
//			//}
//		}
//
//		/*mean *= 1.f/ squared_window_size;*/
//		mean = 1.f / squared_window_size * mean;
//
//		/* compute variance */
//		for (uint i = 0; i < squared_window_size; i++)
//		{
//			uint2 idx = make_uint2((i % window_size + top_left_window_corner.x) % screen.x, (i / window_size + top_left_window_corner.y) % screen.x);
//			float3 input_buffer_val = make_float3(input_buffer[idx].x, input_buffer[idx].y, input_buffer[idx].z);
//			float var = 1.f / 3.f * (input_buffer_val.x + input_buffer_val.y + input_buffer_val.z);
//			/*variance += var * var;*/
//			variance += (var * var - 2.0f * mean * var + mean * mean);
//		}
//
//		//variance = 1.f / squared_window_size * (variance) - (mean * mean);
//		variance = 1.f / squared_window_size * variance;
//
//		per_window_variance_buffer_output[center] = make_float4(variance);
//		//atomicExch(&per_window_variance_buffer_output[center].x, variance);
//
//		rtPrintf("Set variance!!!\n\n");
//	}
//	else
//	{
//		rtPrintf("Reuse variance!!!\n\n");
//		variance = per_window_variance_buffer_output[center].x;
//	}
//
//	return variance;
//};
//
//static __device__ __inline__ uint compute_samples_number(uint2 current_launch_index, float variance)
//{
//	uint samples_number = 0;
//
//	if (adaptive_samples_budget_buffer[current_launch_index].x > 0)
//	{
//		samples_number = static_cast<uint>(clamp(static_cast<float>(variance * max_per_frame_samples_budget), 0.0f, static_cast<float>(max_per_frame_samples_budget)));
//		adaptive_samples_budget_buffer[current_launch_index] = make_int4(adaptive_samples_budget_buffer[current_launch_index].x - static_cast<int>(samples_number));
//	}
//
//	return samples_number;
//};
//
//static __device__ __inline__ uint compute_current_samples_number(uint2 current_launch_index, uint window_size)
//{
//	uint sample_number = 0;
//
//	//uint additional_samples_number = 0;
//
//	size_t2 screen = input_buffer.size();
//
//	uint times_width = screen.x / window_size;
//	uint times_height = screen.y / window_size;
//
//	uint horizontal_padding = static_cast<uint>(0.5f * (screen.x - (times_width * window_size)));
//	uint vertical_padding = static_cast<uint>(0.5f * (screen.y - (times_height * window_size)));
//
//	uint half_window_size = (window_size / 2) + (window_size % 2);
//
//	uint2 times_launch_index = make_uint2(((current_launch_index.x / window_size) * window_size) % screen.x, ((current_launch_index.y / window_size) * window_size) % screen.y);
//
//	uint2 current_window_center = make_uint2(times_launch_index.x + horizontal_padding + half_window_size, times_launch_index.y + vertical_padding + half_window_size);
//
//	float variance = compute_window_variance(current_window_center, window_size);
//
//	sample_number = compute_samples_number(current_launch_index, (30.0f * variance));
//
//	return sample_number;
//};