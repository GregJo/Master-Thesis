/*
* Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <optixu/optixu_math_namespace.h>
#include "optixPathTracer.h"
#include "random.h"
#include "VarianceAdaptive.h"

using namespace optix;

struct PerRayData_pathtrace
{
	float3 result;
	float3 radiance;
	float3 attenuation;
	float3 origin;
	float3 direction;
	unsigned int seed;
	int depth;
	int countEmitted;
	int done;
	//int isAdaptive;
};

struct PerRayData_pathtrace_shadow
{
	bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float, far_plane, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );



//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, rr_begin_depth, , );
rtDeclareVariable(unsigned int, pathtrace_ray_type, , );

// Adaptive post processing variables and buffers

//rtDeclareVariable(unsigned int, window_size, , );
//rtDeclareVariable(unsigned int, max_ray_budget_total, , ) = static_cast<uint>(50u);
//rtDeclareVariable(unsigned int, max_per_frame_samples_budget, , ) = static_cast<uint>(5u);		/* this variable can be written by the user */
//rtDeclareVariable(int, camera_changed, , );

//
// Adaptive version of pathtracing begin
//

//rtDeclareVariable(VarianceAdaptive, variance_adaptive, , );// = VarianceAdaptive();

/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* Adaptive additional rays variables */
//rtDeclareVariable(unsigned int, max_per_frame_samples_budget, , ) = static_cast<uint>(5u);		/* this variable will be written by the user */
//rtBuffer<int4, 2>	  adaptive_samples_budget_buffer;									/* this buffer will be initialized by the host, but must also be modified by the graphics device */
//rtBuffer<int4, 2>	  hoelder_refinement_buffer;										/* this buffer contains the information, where refinement will take place according to
																						//hoelder regularity criterion, everywhere where refinement is needed value is 1, else zero */

rtBuffer<float4, 2>	  hoelder_refinement_buffer;

rtBuffer<int4, 2>	  window_size_buffer;

//rtBuffer<float4, 2>	  per_window_variance_buffer_output;

//rtBuffer<float4, 2>   input_buffer;														/* this buffer contains the initially rendered picture to be post processed */
rtBuffer<float4, 2>   input_scene_depth_buffer;											/* this buffer contains the necessary depth values to compute the gradient 
																						via finite differences for the hoelder alpha computation via the smooth regime */
rtBuffer<float4, 2>   hoelder_adaptive_scene_depth_buffer;								/* this buffer contains only the depth values of the adaptive samples which has been evaluated 
																						and is used for gradient computation */
rtBuffer<float4, 2>   post_process_output_buffer;										/* this buffer contains the result, processed with additional adaptive rays */

// For debug!
rtBuffer<float4, 2>   depth_gradient_buffer;
// For debug!
rtBuffer<float4, 2>   hoelder_alpha_buffer;
// For debug!
rtBuffer<float4, 2>   total_sample_count_buffer;



//
// Hödler Adaptive Image Synthesis (begin)
//

// non-smooth regime
//static __device__ __inline__ float compute_window_hoelder_non_smooth_regime(uint2 center, uint window_size)
//{
//	size_t2 screen = input_buffer.size();
//
//	float alpha = 100.f;
//
//	uint squared_window_size = window_size * window_size;
//	uint half_window_size = (window_size / 2) + (window_size % 2);
//	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);
//
//	//rtPrintf("\nTop left window corner: [ %d, %d ]\n", top_left_window_corner.x, top_left_window_corner.y);
//
//	float3 center_buffer_val = make_float3(input_buffer[center].x, input_buffer[center].y, input_buffer[center].z);
//	float centerColorMean = 1.f / 3.f * (center_buffer_val.x + center_buffer_val.y + center_buffer_val.z);
//	float neighborColorMean = 0.0f;
//
//	/* compute mean value */
//	for (uint i = 0; i < squared_window_size; i++)
//	{
//		uint2 idx = make_uint2((i % window_size + top_left_window_corner.x) % screen.x, (i / window_size + top_left_window_corner.y) % screen.y);
//		if (idx.x == center.x && idx.y == center.y)
//		{
//			continue;
//		}
//		float3 neighbor_buffer_val = make_float3(input_buffer[idx].x, input_buffer[idx].y, input_buffer[idx].z);
//		neighborColorMean = 1.f / 3.f * (neighbor_buffer_val.x + neighbor_buffer_val.y + neighbor_buffer_val.z);
//
//		float neighbor_center_distance = length(make_float2(static_cast<float>(center.x) - static_cast<float>(idx.x), static_cast<float>(center.y) - static_cast<float>(idx.y)));
//
//		float log_base = log(fabs(neighbor_center_distance) + 1.0f);
//
//		if (log_base != 0.0f)
//		{
//			float log_x = log(fabs(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean)) + 1.0f);
//			alpha = min(alpha, (log_x / log_base));
//			alpha = clamp(alpha, 0.0f, 100.f);
//		}
//	}
//	//rtPrintf("___________________________________________________________________________________________\n\n\n");
//
//	return alpha;
//};
//
//// modulo border treatment
//static __device__ __inline__ float3 compute_color_gradient(uint2 idx)
//{
//	size_t2 screen = input_scene_depth_buffer.size();
//
//	uint2 idx_up = make_uint2(idx.x, idx.y + 1 % screen.y);
//	uint2 idx_down = make_uint2(idx.x, min(0, idx.y - 1));//idx.y - 1 < 0 ? screen.y : idx.y - 1);
//
//	//uint2 idx_left = make_uint2(idx.x - 1 < 0 ? screen.x : idx.x - 1, idx.y);
//	uint2 idx_left = make_uint2(min(0, idx.x - 1), idx.y);
//	uint2 idx_right = make_uint2(idx.x + 1 % screen.x, idx.y);
//
//	float4 gradient_y = input_scene_depth_buffer[idx_up] - input_scene_depth_buffer[idx_down];
//	float4 gradient_x = input_scene_depth_buffer[idx_right] - input_scene_depth_buffer[idx_left];
//
//	float4 gradient_tmp = gradient_y + gradient_x;
//
//	float3 gradient = make_float3(gradient_tmp.x / 2.0f, gradient_tmp.y / 2.0f, gradient_tmp.z / 2.0f);
//
//	return gradient;
//};
//
// modulo border treatment
static __device__ __inline__ float3 compute_depth_gradient(uint2 idx)
{
	size_t2 screen = input_buffer.size();

	uint2 idx_up = make_uint2(idx.x, idx.y + 1 % screen.y);
	uint2 idx_down = make_uint2(idx.x, min(0, idx.y - 1));
	uint2 idx_left = make_uint2(min(0, idx.x - 1), idx.y);
	uint2 idx_right = make_uint2(idx.x + 1 % screen.x, idx.y);

	float4 gradient_y = input_buffer[idx_up] - input_buffer[idx_down];
	float4 gradient_x = input_buffer[idx_right] - input_buffer[idx_left];

	float4 gradient_tmp = gradient_y + gradient_x;

	float3 gradient = make_float3(gradient_tmp.x / 2.0f, gradient_tmp.y / 2.0f, gradient_tmp.z / 2.0f);

	return gradient;
};

// modulo border treatment
// first three values of float4 return are the color gradient
// last value of float4 return is the depth/geometry gradient
static __device__ __inline__ float4 compute_color_depth_gradient(uint2 idx)
{
	uint2 screen = make_uint2(input_buffer.size().x, input_buffer.size().y);

	int up = min(idx.y + 1, screen.y);
	int down = max(0, static_cast<int>(idx.y) - 1);
	int left = max(0, static_cast<int>(idx.x) - 1);
	int right = min(idx.x + 1, screen.x);
/*
	if (up > screen.y)
	{
		printf("Up is bigger than screen.y!: %d\n\n", up);
	}
	if (down < 0)
	{
		printf("Down is smaller than 0!: %d\n\n", down);
	}
	if (left < 0)
	{
		printf("Left is smaller than 0!: %d\n\n", left);
	}
	if (right > screen.x)
	{
		printf("Right is bigger than screen.x!: %d\n\n", right);
	}*/

	uint2 idx_up = make_uint2(idx.x, static_cast<uint>(up));
	uint2 idx_down = make_uint2(idx.x, static_cast<uint>(down));
	uint2 idx_left = make_uint2(static_cast<uint>(left), idx.y);
	uint2 idx_right = make_uint2(static_cast<uint>(right), idx.y);

	float4 gradient_color_y = input_buffer[idx_up] - input_buffer[idx_down];
	float4 gradient_color_x = input_buffer[idx_right] - input_buffer[idx_left];

	float4 gradient_color_tmp = gradient_color_y + gradient_color_x;

	float3 gradient_color = make_float3(0.5f * gradient_color_tmp.x, 0.5f * gradient_color_tmp.y, 0.5f * gradient_color_tmp.z);

	float gradient_depth_x = input_scene_depth_buffer[idx_up].x - input_scene_depth_buffer[idx_down].x;
	float gradient_depth_y = input_scene_depth_buffer[idx_right].x - input_scene_depth_buffer[idx_left].x;

	//float gradient_depth_x = hoelder_adaptive_scene_depth_buffer[idx_up].x - hoelder_adaptive_scene_depth_buffer[idx_down].x;
	//float gradient_depth_y = hoelder_adaptive_scene_depth_buffer[idx_right].x - hoelder_adaptive_scene_depth_buffer[idx_left].x;

	float gradient_depth = gradient_depth_x + gradient_depth_y;

	float4 combined_gradient = make_float4(gradient_color.x, gradient_color.y, gradient_color.z, gradient_depth);

	return combined_gradient;
};

static __device__ __inline__ float compute_window_hoelder_smooth_regime(uint2 center, uint window_size)
{
	size_t2 screen = input_buffer.size();

	float alpha = 100.f;

	uint squared_window_size = window_size * window_size;
	uint half_window_size = (window_size / 2) + (window_size % 2);
	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

	float3 center_buffer_val = make_float3(input_buffer[center].x, input_buffer[center].y, input_buffer[center].z);
	float centerColorMean = 1.f / 3.f * (center_buffer_val.x + center_buffer_val.y + center_buffer_val.z);
	float neighborColorMean = 0.0f;

	/* compute mean value */
	for (uint i = 0; i < squared_window_size; i++)
	{
		uint2 idx = make_uint2((i % window_size + top_left_window_corner.x) % screen.x, (i / window_size + top_left_window_corner.y) % screen.y);
		float3 neighbor_buffer_val = make_float3(input_buffer[idx].x, input_buffer[idx].y, input_buffer[idx].z);
		neighborColorMean = 1.f / 3.f * (neighbor_buffer_val.x + neighbor_buffer_val.y + neighbor_buffer_val.z);

		/*float gradient_of_mean_color = length(compute_color_gradient(idx));*/
		float gradient_of_mean_color = length(compute_depth_gradient(idx));

		float neighbor_center_distance = length(make_float2(static_cast<float>(center.x) - static_cast<float>(idx.x), static_cast<float>(center.y) - static_cast<float>(idx.y)));

		float log_base = log(fabs(neighbor_center_distance) + 1.0f);

		if (log_base != 0.0f)
		{
			float log_x = log(fabs(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean - gradient_of_mean_color * neighbor_center_distance) + 1.0f));

			alpha = min(alpha, log_x / log_base);

			alpha = clamp(alpha, 0.0f, 100.f);
		}
	}

	return alpha;
};

static __device__ __inline__ float compute_window_hoelder(uint2 center, uint window_size)
{
	size_t2 screen = input_buffer.size();

	float alpha = 100.f;

	uint squared_window_size = window_size * window_size;
	uint half_window_size = (window_size / 2) + (window_size % 2);
	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

	float3 center_buffer_val = make_float3(input_buffer[center].x, input_buffer[center].y, input_buffer[center].z);
	float centerColorMean = 1.f / 3.f * (center_buffer_val.x + center_buffer_val.y + center_buffer_val.z);
	float neighborColorMean = 0.0f;

	for (uint i = 0; i < squared_window_size; i++)
	{
		uint2 idx = make_uint2((i % window_size + top_left_window_corner.x) % screen.x, (i / window_size + top_left_window_corner.y) % screen.y);

		float4 color_depth_gradient = compute_color_depth_gradient(idx);

		// Debug!
		depth_gradient_buffer[idx] = make_float4(color_depth_gradient.w);

		float neighbor_center_distance = length(make_float2(static_cast<float>(center.x) - static_cast<float>(idx.x), static_cast<float>(center.y) - static_cast<float>(idx.y)));

		float log_base = log(fabsf(neighbor_center_distance) + 1.0f);

		//if (i % window_size <= i / window_size)
		//{
		//	float inverseWindowSize = 1.0f / static_cast<float>(window_size);
		//	post_process_output_buffer[idx] = make_float4(window_size * inverseWindowSize, 0.0f, window_size_buffer[idx].x * inverseWindowSize, 1.0f);
		//}

		//if (log_base == 0.0f)
		//{
		//	rtPrintf("Neighbor center distance: || [ %f , %f ]-[ %f , %f ] || = %f \n\n", 
		//		static_cast<float>(center.x), static_cast<float>(center.y), 
		//		static_cast<float>(idx.x), static_cast<float>(idx.y), 
		//		neighbor_center_distance);
		//}

		if (log_base != 0.0f)
		{
			float3 neighbor_buffer_val = make_float3(input_buffer[idx].x, input_buffer[idx].y, input_buffer[idx].z);
			neighborColorMean = 1.f / 3.f * (neighbor_buffer_val.x + neighbor_buffer_val.y + neighbor_buffer_val.z);
			//float log_x = log(fabs(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean - gradient_of_mean_color * neighbor_center_distance) + 1.0f));
			float log_x = 0.0f;

			// Decide whether to use smooth or non-smooth regime based on depth/geometry buffer map. 
			// Where there is a very small depth/geometry gradient use smooth regime computation hoelder alpha, 
			// else use non-smooth regime hoelder alpha computation (log_x value makes for that distinction). 
			if (fabsf(color_depth_gradient.w)/* Value 'w' is depth/geometry gradient */ <= 0.01f/* Currently more or less arbitary threshhold for an edge! */)
			{
				//post_process_output_buffer[idx] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
				//rtPrintf("\nsmooth!!!\n");
				float3 color_gradient = make_float3(color_depth_gradient.x, color_depth_gradient.y, color_depth_gradient.z);
				float mean_of_color_gradient = length(color_gradient);
				log_x = log(fabsf(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean - mean_of_color_gradient * neighbor_center_distance)) + 1.0f);
			}
			else
			{
				//post_process_output_buffer[idx] = make_float4(100.0f, 0.0f, 100.0f, 1.0f);
				//rtPrintf("\nnon-smooth!!!\n");
				float log_x = log(fabsf(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean)) + 1.0f);
			}

			//if (idx.x == 0)
			//{
			//	rtPrintf("Temporary alpha: %f\n\n", (log_x / log_base));
			//}
			//if (idx.x == 1 && idx.y == 1)
			//{
			//	rtPrintf("\n\n");
			//}
			

			alpha = min(alpha, log_x / log_base);
			alpha = clamp(alpha, 0.0f, 100.f);
		}
	}

	return alpha;
};

static __device__ __inline__ float4 hoelder_refinement(float alpha, uint2 center, uint window_size)
{
	float4 alphas = make_float4(0.0f);

	uint2 center1 = center + make_uint2((center.x - 0.5 * center.x), (center.y - 0.5 * center.y));
	uint2 center2 = center + make_uint2((center.x + 0.5 * center.x), (center.y - 0.5 * center.y));
	uint2 center3 = center + make_uint2((center.x - 0.5 * center.x), (center.y + 0.5 * center.y));
	uint2 center4 = center + make_uint2((center.x + 0.5 * center.x), (center.y + 0.5 * center.y));

	uint half_window_size = 0.5f * window_size;

	if (alpha < 0.5f /* Arbitary alpha threshold */)
	{
		alphas.x = compute_window_hoelder(center1, half_window_size);
		alphas.y = compute_window_hoelder(center2, half_window_size);
		alphas.z = compute_window_hoelder(center3, half_window_size);
		alphas.w = compute_window_hoelder(center4, half_window_size);
	}

	return alphas;
}

// TODO: Rename this function to something more general, like "expend_samples_of_sample_map"
static __device__ __inline__ uint compute_hoelder_samples_number(uint2 current_launch_index, uint window_size)
{
	//rtPrintf("Hoelder alpha: %f\n\n", alpha);

	//float oversampling_factor = 1.25f;

	uint samples_number = min(adaptive_samples_budget_buffer[current_launch_index].x, max_per_frame_samples_budget);

	//rtPrintf("Currently avaible adaptive samples: %d\n\n", samples_number);

	if (adaptive_samples_budget_buffer[current_launch_index].x > 0)
	{
		//samples_number = static_cast<uint>(clamp(static_cast<float>(samples_number), 0.0f, static_cast<float>(max_per_frame_samples_budget)));
		adaptive_samples_budget_buffer[current_launch_index] = make_int4(adaptive_samples_budget_buffer[current_launch_index].x - static_cast<int>(samples_number));
	}

	//rtPrintf("Currently avaible adaptive samples: %d\n\n", adaptive_samples_budget_buffer[current_launch_index].x);

	return samples_number;
};

static __device__ __inline__ uint hoelder_compute_current_samples_number_and_manage_buffers(uint2 current_launch_index, uint2 current_window_center, uint window_size)
{
	if (window_size >= 2)
	{
		float hoelder_alpha = -1.0f;
		float hoelder_alpha_no_refinement_threshhold = 0.5f;

		//rtPrintf("Hoelder refinement buffer value: %d\n", hoelder_refinement_buffer[current_launch_index].x);
		//if (adaptive_samples_budget_buffer[current_launch_index].x == 1)
		//{
		//	rtPrintf("Currently avaible adaptive samples: %d\n\n", adaptive_samples_budget_buffer[current_launch_index].x);
		//}
		/*rtPrintf("Currently avaible adaptive samples: %d\n\n", adaptive_samples_budget_buffer[current_launch_index].x);*/

		if (hoelder_refinement_buffer[current_launch_index].x == 1)
		{
			//rtPrintf("Compute!!!\n\n");
			hoelder_alpha = compute_window_hoelder(current_window_center, window_size);
			//hoelder_refinement_buffer[current_launch_index] = make_int4(0);
			hoelder_refinement_buffer[current_launch_index] = make_float4(0);

			//printf("Current hoelder: %f\n\n", hoelder_alpha);
		}

		//rtPrintf("Current hoelder: %f\n\n", hoelder_alpha);

		hoelder_alpha_buffer[current_launch_index] = make_float4(hoelder_alpha * 10.0f);

		if (hoelder_alpha < 0.0f)
		{
			//rtPrintf("Set to one hundred!!!\n\n");
			hoelder_alpha = 100.0f;
		}

		if (hoelder_alpha * 100.0f < hoelder_alpha_no_refinement_threshhold)
		{
			//printf("Current hoelder: %f\n\n", hoelder_alpha);
			//rtPrintf("Refine next frame!!!\n\n");
			//hoelder_refinement_buffer[current_launch_index] = make_int4(1);
			hoelder_refinement_buffer[current_launch_index] = make_float4(1);
			adaptive_samples_budget_buffer[current_launch_index] += make_int4(1);// hoelder_refinement_buffer[current_launch_index];
			//total_sample_count_buffer[current_launch_index] += make_float4(1.0f/log2f(static_cast<float>(window_size) + 1));
			window_size_buffer[current_launch_index] = make_int4(0.5f * window_size_buffer[current_launch_index].x);
			//rtPrintf("Currently avaible adaptive samples: %d\n\n", adaptive_samples_budget_buffer[current_launch_index].x);
		}
	}

	return compute_hoelder_samples_number(current_launch_index, window_size);
	//return 0;
};

static __device__ __inline__ void initialize_hoelder_refinement_buffer(uint2 current_launch_index, int frame_number, int camera_changed, uint window_size)
{
	if (frame_number == 1 || camera_changed == 1)
	{
		//rtPrintf("Initialize holder refinement buffer!!!\n\n");
		//hoelder_refinement_buffer[current_launch_index] = make_int4(1);
		hoelder_refinement_buffer[current_launch_index] = make_float4(1);
		adaptive_samples_budget_buffer[current_launch_index] = make_int4(0);
		total_sample_count_buffer[current_launch_index] = make_float4(0.0f);

		window_size_buffer[current_launch_index] = make_int4(window_size);
		//rtPrintf("Currently avaible adaptive samples: %d\n\n", adaptive_samples_budget_buffer[current_launch_index].x);
	}
};

static __device__ __inline__ void initializeHoelderAdaptiveSceneDepthBuffer(uint2 current_launch_index, int frame_number, int camera_changed)
{
	if (frame_number == 1 || camera_changed == 1)
	{
		//rtPrintf("Init!!!\n\n");
		hoelder_adaptive_scene_depth_buffer[current_launch_index] = input_scene_depth_buffer[current_launch_index];
	}
};

static __device__ __inline__ void resetHoelderAdaptiveSceneDepthBuffer(uint2 current_launch_index)
{
	hoelder_adaptive_scene_depth_buffer[current_launch_index] = make_float4(0.0f);
};

//
// Hödler Adaptive Image Synthesis (end)
//

static __device__ __inline__ uint compute_current_samples_number(uint2 current_launch_index, uint window_size)
{
	uint sample_number = 0;

	//uint additional_samples_number = 0;

	size_t2 screen = input_buffer.size();

	uint times_width = screen.x / window_size;
	uint times_height = screen.y / window_size;

	uint horizontal_padding = static_cast<uint>(0.5f * (screen.x - (times_width * window_size)));
	uint vertical_padding = static_cast<uint>(0.5f * (screen.y - (times_height * window_size)));

	uint half_window_size = (window_size / 2) + (window_size % 2);

	uint2 times_launch_index = make_uint2(((current_launch_index.x / window_size) * window_size) % screen.x, ((current_launch_index.y / window_size) * window_size) % screen.y);

	uint2 current_window_center = make_uint2(times_launch_index.x + horizontal_padding + half_window_size, times_launch_index.y + vertical_padding + half_window_size);

	//float variance = compute_window_variance(current_window_center, window_size);

	//float hoelder_alpha = compute_window_hoelder(current_window_center, window_size);

	//hoelder_alpha_buffer[current_launch_index] = make_float4(hoelder_alpha * 100.0f);

	//sample_number = compute_samples_number(current_launch_index, (30.0f * variance));

	//sample_number = compute_hoelder_samples_number(current_launch_index, (10.0f * hoelder_alpha), window_size);

	sample_number = hoelder_compute_current_samples_number_and_manage_buffers(current_launch_index, current_window_center, window_size);
	float sample_count_fraction = static_cast<float>(sample_number) / log2f(static_cast<float>(window_size) + 50);
	total_sample_count_buffer[current_launch_index] += make_float4(sample_count_fraction, 0.0f, 0.0f, 1.0f);

	//hoelder_refinement(hoelder_alpha, current_window_center, window_size);

	rtPrintf("Sample number: %d\n\n", sample_number);

	return sample_number;
};

//static __device__ __inline__ void init_variance_adaptive_struct(VarianceAdaptiveStruct* variance_adaptive_struct)
//{
//	variance_adaptive_struct->_adaptive_samples_budget_buffer = &adaptive_samples_budget_buffer;
//	variance_adaptive_struct->_input_buffer = &input_buffer;
//	variance_adaptive_struct->_per_window_variance_buffer_output = &per_window_variance_buffer_output;
//
//	variance_adaptive_struct->_max_per_frame_samples_budget = max_per_frame_samples_budget;
//	variance_adaptive_struct->_window_size = window_size;
//};

RT_PROGRAM void pathtrace_camera_adaptive()
{
	//rtPrintf("Current samples number: %d\n\n", adaptive_samples_budget_buffer[launch_index].x);

	// Debug!
	depth_gradient_buffer[launch_index] = make_float4(0.0f);

	size_t2 screen = input_buffer.size();

	float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
	float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	float2 jitter_scale = inv_screen / sqrt_num_samples;

	//initializeHoelderAdaptiveSceneDepthBuffer(launch_index, frame_number, camera_changed);
	//initialize_hoelder_refinement_buffer(launch_index, frame_number, camera_changed, static_cast<uint>(window_size));

	//VarianceAdaptiveStruct variance_adaptive_struct;
	//init_variance_adaptive_struct(&variance_adaptive_struct);

	//unsigned int adaptive_samples_per_pixel = compute_current_samples_number(launch_index, variance_adaptive_struct);//, window_size, max_per_frame_samples_budget);//compute_current_samples_number(launch_index, window_size_buffer[launch_index].x, 
																			//&input_buffer, 
																			//&adaptive_samples_budget_buffer, max_per_frame_samples_budget, &per_window_variance_buffer_output);
	unsigned int adaptive_samples_per_pixel = compute_current_samples_number(launch_index, window_size);
	unsigned int current_samples_per_pixel = adaptive_samples_per_pixel;
	float3 result = make_float3(0.0f);

	unsigned int adaptive_sqrt_num_samples = sqrtf(static_cast<float>(adaptive_samples_per_pixel));

	if (!adaptive_sqrt_num_samples)
	{
		++adaptive_sqrt_num_samples;
	}

	unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x, frame_number);

	float3 pixel_color = make_float3(input_buffer[launch_index]);

	//resetHoelderAdaptiveSceneDepthBuffer(launch_index);

	if (current_samples_per_pixel)
	{
		do
		{
			//
			// Sample pixel using jittering
			//
			unsigned int x = adaptive_samples_per_pixel % adaptive_sqrt_num_samples;
			unsigned int y = adaptive_samples_per_pixel / adaptive_sqrt_num_samples;
			float2 jitter = make_float2(x - rnd(seed), y - rnd(seed));
			float2 d = pixel + jitter*jitter_scale;
			float3 ray_origin = eye;
			float3 ray_direction = normalize(d.x*U + d.y*V + W);

			// Initialze per-ray data
			PerRayData_pathtrace prd;
			prd.result = make_float3(0.f);
			prd.attenuation = make_float3(1.f);
			prd.countEmitted = true;
			prd.done = false;
			prd.seed = seed;
			prd.depth = 0;
			//prd.isAdaptive = 1;

			// Each iteration is a segment of the ray path.  The closest hit will
			// return new segments to be traced here.
			for (;;)
			{
				if (prd.depth == 1)
				{
					float ray_length = fabsf(length((prd.origin - eye)));
					float normalized_ray_length = ray_length / far_plane;//2500.0f;

					float a = 1.0f / (float)frame_number;
					float3 old_depth = make_float3(input_scene_depth_buffer[launch_index]);
					input_scene_depth_buffer[launch_index] = make_float4(lerp(old_depth, make_float3(normalized_ray_length), a), 1.0f);

					hoelder_adaptive_scene_depth_buffer[launch_index] = make_float4(make_float3(normalized_ray_length), 1.0f);
					//if (frame_number == 1)
					//{
					//	input_scene_depth_buffer[launch_index] = make_float4(normalized_ray_length);
					//}
				}

				//if (prd.depth == 1)
				//{
				//	float ray_length = fabsf(length((prd.origin - eye)));
				//	float normalized_ray_length = ray_length / far_plane;//2500.0f;

				//	hoelder_adaptive_scene_depth_buffer[launch_index] = make_float4(make_float3(normalized_ray_length), 1.0f);
				//}

				Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
				rtTrace(top_object, ray, prd);

				if (prd.done)
				{
					// We have hit the background or a luminaire
					prd.result += prd.radiance * prd.attenuation;
					break;
				}

				// Russian roulette termination 
				if (prd.depth >= rr_begin_depth)
				{
					float pcont = fmaxf(prd.attenuation);
					if (rnd(prd.seed) >= pcont)
						break;
					prd.attenuation /= pcont;
				}

				prd.depth++;
				prd.result += prd.radiance * prd.attenuation;

				// Update ray data for the next path segment
				ray_origin = prd.origin;
				ray_direction = prd.direction;
			}

			result += prd.result;
			seed = prd.seed;
		} while (--current_samples_per_pixel);

		pixel_color = result / (adaptive_sqrt_num_samples*adaptive_sqrt_num_samples);

		// Pink coloring of tiles for debug
		//if (adaptive_samples_per_pixel == 1 && window_size_buffer[launch_index].x <= 4)
		//{
		//	pixel_color = make_float3(window_size, 0.0f, window_size_buffer[launch_index].x);
		//}

		//if (adaptive_samples_per_pixel >= 1)
		//{
		//	pixel_color = make_float3(0.0f);
		//}
	}
	//
	// Update the output buffer
	//

	float a = 1.0f / (float)frame_number;
	float3 old_color = make_float3(input_buffer[launch_index]);
	post_process_output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a), 1.0f);

	//compute_current_window_test(launch_index, 5);
}

//
// Adaptive version of pathtracing end
//
