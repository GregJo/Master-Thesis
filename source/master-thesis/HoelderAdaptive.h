#include <optixu/optixu_math_namespace.h>
#include <optix_device.h>

#include "Buffers.h"

//
// A way to implement and use classes in cuda code
//
//#ifdef __CUDACC__
//#define CUDA_CALLABLE_MEMBER __host__ __device__
//#else
//#define CUDA_CALLABLE_MEMBER
//#endif 

//enum HoelderAdaptiveBufferIDs 
//{
//	input_scene_render_buffer = 0,
//	//input_scene_depth_buffer = 1,
//	
//	//hoelder_adaptive_scene_depth_buffer = 1,
//	//hoelder_refinement_buffer = 3,
//
//	//adaptive_samples_budget_buffer = 2,
//	//hoelder_window_size_buffer = 5
//};

using namespace optix;

rtDeclareVariable(unsigned int, window_size, , );
rtDeclareVariable(unsigned int, initial_window_size, , );
rtDeclareVariable(unsigned int, current_level_adaptive_sample_count, , );
rtDeclareVariable(unsigned int, max_per_frame_samples_budget, , ) = static_cast<uint>(4u);		/* this variable can be written by the user */
rtDeclareVariable(int, camera_changed, , );


//rtBuffer<rtBufferId<float4,2>, 1>	  hoelder_adaptive_buffers;									/* this buffer will be initialized by the host, but must also be modified by the graphics device */


rtBuffer<int4, 2>	  adaptive_samples_budget_buffer;									/* this buffer will be initialized by the host, but must also be modified by the graphics device */

rtBuffer<float4, 2>	  hoelder_refinement_buffer;

// Currently testing
rtBuffer<float4, 2>	  hoelder_level_output_buffer;

rtBuffer<int4, 2>	  window_size_buffer;

//rtBuffer<int4, 2>	  current_max_adaptive_level_samples_number;

//rtBuffer<float4, 2>   input_buffer;														/* this buffer contains the initially rendered picture to be post processed */
//rtBuffer<float4, 2>   input_scene_depth_buffer;											/* this buffer contains the necessary depth values to compute the gradient
																						//via finite differences for the hoelder alpha computation via the smooth regime */
rtBuffer<float4, 2>   hoelder_adaptive_scene_depth_buffer;								/* this buffer contains only the depth values of the adaptive samples which has been evaluated
																						//and is used for gradient computation */

// Currently testing
rtBuffer<float4, 2>   hoelder_adaptive_level_scene_depth_buffer;

// For debug!
rtBuffer<float4, 2>   depth_gradient_buffer;
// For debug!
rtBuffer<float4, 2>   hoelder_alpha_buffer;
// For debug!
rtBuffer<float4, 2>   total_sample_count_buffer;

// modulo border treatment
// first three values of float4 return are the color gradient
// last value of float4 return is the depth/geometry gradient
static __device__ __inline__ float4 compute_color_depth_gradient(uint2 idx)
{
	//uint2 screen = make_uint2(hoelder_adaptive_buffers[input_scene_render_buffer].size().x, hoelder_adaptive_buffers[input_scene_render_buffer].size().y);
	uint2 screen = make_uint2(output_buffer.size().x, output_buffer.size().y);

	int up = min(idx.y + 1, screen.y);
	int down = max(0, static_cast<int>(idx.y) - 1);
	int left = max(0, static_cast<int>(idx.x) - 1);
	int right = min(idx.x + 1, screen.x);

	uint2 idx_up = make_uint2(idx.x, static_cast<uint>(up));
	uint2 idx_down = make_uint2(idx.x, static_cast<uint>(down));
	uint2 idx_left = make_uint2(static_cast<uint>(left), idx.y);
	uint2 idx_right = make_uint2(static_cast<uint>(right), idx.y);

	//float4 gradient_color_y = hoelder_adaptive_buffers[input_scene_render_buffer][idx_up] - hoelder_adaptive_buffers[input_scene_render_buffer][idx_down];
	//float4 gradient_color_x = hoelder_adaptive_buffers[input_scene_render_buffer][idx_right] - hoelder_adaptive_buffers[input_scene_render_buffer][idx_left];

	float4 gradient_color_y = output_buffer[idx_up] - output_buffer[idx_down];
	float4 gradient_color_x = output_buffer[idx_right] - output_buffer[idx_left];

	float4 gradient_color_tmp = gradient_color_y + gradient_color_x;

	float3 gradient_color = make_float3(0.5f * gradient_color_tmp.x, 0.5f * gradient_color_tmp.y, 0.5f * gradient_color_tmp.z);

	float gradient_depth_x = output_scene_depth_buffer[idx_up].x - output_scene_depth_buffer[idx_down].x;
	float gradient_depth_y = output_scene_depth_buffer[idx_right].x - output_scene_depth_buffer[idx_left].x;

	float gradient_depth = gradient_depth_x + gradient_depth_y;

	float4 combined_gradient = make_float4(gradient_color.x, gradient_color.y, gradient_color.z, gradient_depth);

	return combined_gradient;
};

// Currently testing
static __device__ __inline__ float4 compute_level_color_depth_gradient(uint2 current_launch_index, uint window_size_)
{
	uint2 screen = make_uint2(output_buffer.size().x, output_buffer.size().y);

	//rtPrintf("Output buffer size: [ %u , %u ]\n\n", screen.x, screen.y);

	int up = min((static_cast<int>(current_launch_index.y) + static_cast<int>(window_size_)), static_cast<int>(screen.y));
	int down = max(0, (static_cast<int>(current_launch_index.y) - static_cast<int>(window_size_)));
	int left = max(0, (static_cast<int>(current_launch_index.x) - static_cast<int>(window_size_)));
	int right = min((static_cast<int>(current_launch_index.x) + static_cast<int>(window_size_)), static_cast<int>(screen.x));

	//rtPrintf("[ up , down , left , right ]: [ %d , %d , %d , %d ]\n\n", up, down, left, right);

	uint2 idx_up = make_uint2(current_launch_index.x, static_cast<uint>(up));
	uint2 idx_down = make_uint2(current_launch_index.x, static_cast<uint>(down));
	uint2 idx_left = make_uint2(static_cast<uint>(left), current_launch_index.y);
	uint2 idx_right = make_uint2(static_cast<uint>(right), current_launch_index.y);

	float4 gradient_color_y = output_buffer[idx_up] - output_buffer[idx_down];
	float4 gradient_color_x = output_buffer[idx_right] - output_buffer[idx_left];

	float4 gradient_color_tmp = gradient_color_y + gradient_color_x;

	float3 gradient_color = make_float3(0.5f * gradient_color_tmp.x, 0.5f * gradient_color_tmp.y, 0.5f * gradient_color_tmp.z);

	float gradient_depth_x = output_scene_depth_buffer[idx_up].x - output_scene_depth_buffer[idx_down].x;
	float gradient_depth_y = output_scene_depth_buffer[idx_right].x - output_scene_depth_buffer[idx_left].x;

	float gradient_depth = gradient_depth_x + gradient_depth_y;

	float4 combined_gradient = make_float4(gradient_color.x, gradient_color.y, gradient_color.z, gradient_depth);

	return combined_gradient;
};

static __device__ __inline__ float compute_window_hoelder(uint2 center, uint window_size_)
{
	//size_t2 screen = hoelder_adaptive_buffers[input_scene_render_buffer].size();
	size_t2 screen = output_buffer.size();

	float alpha = 100.f;

	uint squared_window_size = window_size_ * window_size_;
	uint half_window_size = (window_size_ / 2) + (window_size_ % 2);
	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

	//float3 center_buffer_val = make_float3(hoelder_adaptive_buffers[input_scene_render_buffer][center].x, hoelder_adaptive_buffers[input_scene_render_buffer][center].y, hoelder_adaptive_buffers[input_scene_render_buffer][center].z);
	float3 center_buffer_val = make_float3(output_buffer[center].x, output_buffer[center].y, output_buffer[center].z);

	float centerColorMean = 1.f / 3.f * (center_buffer_val.x + center_buffer_val.y + center_buffer_val.z);
	float neighborColorMean = 0.0f;

	for (uint i = 0; i < squared_window_size; i++)
	{
		uint2 idx = make_uint2((i % window_size_ + top_left_window_corner.x) % screen.x, (i / window_size_ + top_left_window_corner.y) % screen.y);

		float4 color_depth_gradient = compute_color_depth_gradient(idx);

		// Debug!
		depth_gradient_buffer[idx] = make_float4(color_depth_gradient.w);

		float neighbor_center_distance = length(make_float2(static_cast<float>(center.x) - static_cast<float>(idx.x), static_cast<float>(center.y) - static_cast<float>(idx.y)));

		float log_base = log(fabsf(neighbor_center_distance) + 1.0f);

		if (log_base != 0.0f)
		{
			/*float3 neighbor_buffer_val = make_float3(hoelder_adaptive_buffers[input_scene_render_buffer][idx].x, hoelder_adaptive_buffers[input_scene_render_buffer][idx].y, hoelder_adaptive_buffers[input_scene_render_buffer][idx].z);*/
			float3 neighbor_buffer_val = make_float3(output_buffer[idx].x, output_buffer[idx].y, output_buffer[idx].z);
			neighborColorMean = 1.f / 3.f * (neighbor_buffer_val.x + neighbor_buffer_val.y + neighbor_buffer_val.z);
			float log_x = 0.0f;

			// Decide whether to use smooth or non-smooth regime based on depth/geometry buffer map. 
			// Where there is a very small depth/geometry gradient use smooth regime computation hoelder alpha, 
			// else use non-smooth regime hoelder alpha computation (log_x value makes for that distinction). 
			if (fabsf(color_depth_gradient.w)/* Value 'w' is depth/geometry gradient */ <= 0.01f/* Currently more or less arbitary threshhold for an edge! */)
			{
				float3 color_gradient = make_float3(color_depth_gradient.x, color_depth_gradient.y, color_depth_gradient.z);
				float mean_of_color_gradient = length(color_gradient);
				log_x = log(fabsf(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean - mean_of_color_gradient * neighbor_center_distance)) + 1.0f);
			}
			else
			{
				float log_x = log(fabsf(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean)) + 1.0f);
			}

			alpha = min(alpha, log_x / log_base);
			alpha = clamp(alpha, 0.0f, 100.f);
		}
	}

	return alpha;
};

// Currently testing
static __device__ __inline__ float compute_level_window_hoelder(uint2 center, uint window_size_)
{
	//size_t2 screen = hoelder_adaptive_buffers[input_scene_render_buffer].size();
	uint2 screen = make_uint2(output_buffer.size().x, output_buffer.size().y);

	float alpha = 100.f;

	uint squared_window_size = window_size_ * window_size_;
	uint half_window_size = (window_size_ / 2) + (window_size_ % 2);
	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

	//float3 center_buffer_val = make_float3(hoelder_adaptive_buffers[input_scene_render_buffer][center].x, hoelder_adaptive_buffers[input_scene_render_buffer][center].y, hoelder_adaptive_buffers[input_scene_render_buffer][center].z);
	float3 center_buffer_val = make_float3(output_buffer[center].x, output_buffer[center].y, output_buffer[center].z);

	float centerColorMean = 1.f / 3.f * (center_buffer_val.x + center_buffer_val.y + center_buffer_val.z);
	float neighborColorMean = 0.0f;

	float4 color_depth_gradient = compute_level_color_depth_gradient(center, window_size_); //compute_color_depth_gradient(center);//

	// Debug!
	depth_gradient_buffer[center] = make_float4(color_depth_gradient.w);

	for (size_t i = 0; i < 8; i++)
	{
		uint2 idx = make_uint2(0);

		if (i == 0)
		{
			int up = min((static_cast<int>(center.y) + static_cast<int>(window_size_)), static_cast<int>(screen.y));
			idx = make_uint2(center.x, static_cast<uint>(up));
		}
		if (i == 1)
		{
			int down = max(0, (static_cast<int>(center.y) - static_cast<int>(window_size_)));
			idx = make_uint2(center.x, static_cast<uint>(down));
		}
		if (i == 2)
		{
			int left = max(0, (static_cast<int>(center.x) - static_cast<int>(window_size_)));
			idx = make_uint2(static_cast<uint>(left), center.y);
		}
		if (i == 3)
		{
			int right = min((static_cast<int>(center.x) + static_cast<int>(window_size_)), static_cast<int>(screen.x));
			idx = make_uint2(static_cast<uint>(right), center.y);
		}

		if (i == 4)
		{
			int top_left_x = max(0, (static_cast<int>(center.x) - static_cast<int>(window_size_)));
			int top_left_y = min((static_cast<int>(center.y) + static_cast<int>(window_size_)), screen.y);
			idx = make_uint2(static_cast<uint>(top_left_x), static_cast<uint>(top_left_y));
		}
		if (i == 5)
		{
			int top_right_x = min((static_cast<int>(center.x) + static_cast<int>(window_size_)), static_cast<int>(screen.x));
			int top_right_y = min((static_cast<int>(center.y) + static_cast<int>(window_size_)), static_cast<int>(screen.y));
			idx = make_uint2(static_cast<uint>(top_right_x), static_cast<uint>(top_right_y));
		}
		if (i == 6)
		{
			int bottom_left_x = max(0, (static_cast<int>(center.x) - static_cast<int>(window_size_)));
			int bottom_left_y = max(0, (static_cast<int>(center.y) - static_cast<int>(window_size_)));
			idx = make_uint2(static_cast<uint>(bottom_left_x), static_cast<uint>(bottom_left_y));
		}
		if (i == 7)
		{
			int bottom_right_x = min((static_cast<int>(center.x) + static_cast<int>(window_size_)), static_cast<int>(screen.x));
			int bottom_right_y = max(0, (static_cast<int>(center.y) - static_cast<int>(window_size_)));
			idx = make_uint2(static_cast<uint>(bottom_right_x), static_cast<uint>(bottom_right_y));
		}

		float neighbor_center_distance = length(make_float2(static_cast<float>(center.x) - static_cast<float>(window_size), static_cast<float>(center.y) - static_cast<float>(window_size_)));

		float log_base = log(fabsf(neighbor_center_distance) + 1.0f);

		if (log_base != 0.0f)
		{
			/*float3 neighbor_buffer_val = make_float3(hoelder_adaptive_buffers[input_scene_render_buffer][idx].x, hoelder_adaptive_buffers[input_scene_render_buffer][idx].y, hoelder_adaptive_buffers[input_scene_render_buffer][idx].z);*/
			float3 neighbor_buffer_val = make_float3(output_buffer[idx].x, output_buffer[idx].y, output_buffer[idx].z);
			neighborColorMean = 1.f / 3.f * (neighbor_buffer_val.x + neighbor_buffer_val.y + neighbor_buffer_val.z);
			float log_x = 0.0f;

			// Decide whether to use smooth or non-smooth regime based on depth/geometry buffer map. 
			// Where there is a very small depth/geometry gradient use smooth regime computation hoelder alpha, 
			// else use non-smooth regime hoelder alpha computation (log_x value makes for that distinction). 
			if (fabsf(color_depth_gradient.w)/* Value 'w' is depth/geometry gradient */ <= 0.01f/* Currently more or less arbitary threshhold for an edge! */)
			{
				float3 color_gradient = make_float3(color_depth_gradient.x, color_depth_gradient.y, color_depth_gradient.z);
				float mean_of_color_gradient = length(color_gradient);
				log_x = log(fabsf(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean - mean_of_color_gradient * neighbor_center_distance)) + 1.0f);
			}
			else
			{
				float log_x = log(fabsf(1.0f / 2.0f /*hoelder constant, also try value 3*/ * (centerColorMean - neighborColorMean)) + 1.0f);
			}

			alpha = min(alpha, log_x / log_base);
			alpha = clamp(alpha, 0.0f, 100.f);
		}
	}

	return alpha;
};

// TODO: Rename this function to something more general, like "expend_samples_of_sample_map"
static __device__ __inline__ uint compute_hoelder_samples_number(uint2 current_launch_index)
{
	uint samples_number = min(adaptive_samples_budget_buffer[current_launch_index].x, max_per_frame_samples_budget);

	if (adaptive_samples_budget_buffer[current_launch_index].x > 0)
	{
		adaptive_samples_budget_buffer[current_launch_index] = make_int4(adaptive_samples_budget_buffer[current_launch_index].x - static_cast<int>(samples_number));
	}

	return samples_number;
};

static __device__ __inline__ uint hoelder_compute_current_samples_number_and_manage_buffers(uint2 current_launch_index, uint2 current_window_center, uint window_size_)
{
	if (adaptive_samples_budget_buffer[current_launch_index].x <= 0)
	{
		//if (current_launch_index.x == 128 && current_launch_index.y == 128)
		//{
		//	rtPrintf("Adaptive samples budget buffer(device): %d\n", adaptive_samples_budget_buffer[current_launch_index].x);
		//}
		if (window_size_ >= 1)
		{
			//if (current_launch_index.x == 128 && current_launch_index.y == 128)
			//{
			//	rtPrintf("Current window size(device): %d\n", window_size_);
			//	rtPrintf("Current level adaptive sample count(device): %d\n", current_level_adaptive_sample_count);
			//}
			float hoelder_alpha = -1.0f;
			float hoelder_alpha_no_refinement_threshhold = 0.2f;//0.05f

			if (hoelder_refinement_buffer[current_launch_index].x == 1)
			{
				if (hoelder_adaptive_scene_depth_buffer[current_launch_index].x < 1.0f)
				{
					hoelder_alpha = compute_level_window_hoelder(current_window_center, window_size_);//compute_window_hoelder(current_window_center, window_size_);
					//hoelder_alpha = compute_window_hoelder(current_window_center, window_size_);
				}
				hoelder_refinement_buffer[current_launch_index] = make_float4(0);
			}

			hoelder_alpha_buffer[current_launch_index] = make_float4(hoelder_alpha * 10.0f);

			if (hoelder_alpha < 0.0f)
			{
				hoelder_alpha = 10.0f;
			}

			// Luminosity conversion: 0.21 R + 0.72 G + 0.07 B
			//float currentPixelLuminosity = 0.21f * hoelder_adaptive_buffers[input_scene_render_buffer][current_launch_index].x + 0.72f * hoelder_adaptive_buffers[input_scene_render_buffer][current_launch_index].y + 0.07f * hoelder_adaptive_buffers[input_scene_render_buffer][current_launch_index].z;
			float currentPixelLuminosity = 0.21f * output_buffer[current_launch_index].x + 0.72f * output_buffer[current_launch_index].y + 0.07f * output_buffer[current_launch_index].z;
			float overSampleFactor = 1.25f;

			if (hoelder_alpha * 10.0f < hoelder_alpha_no_refinement_threshhold)// && currentPixelLuminosity > 0.0f)// * currentPixelLuminosity)
			{
				hoelder_refinement_buffer[current_launch_index] = make_float4(1);
				adaptive_samples_budget_buffer[current_launch_index] += make_int4(overSampleFactor * current_level_adaptive_sample_count);//make_int4(1);// hoelder_refinement_buffer[current_launch_index];
																													   //total_sample_count_buffer[current_launch_index] += make_float4(1.0f/log2f(static_cast<float>(window_size) + 1));
				window_size_buffer[current_launch_index] = make_int4(0.5f * window_size_buffer[current_launch_index].x);
			}
		}

		//current_max_adaptive_level_samples_number[current_launch_index].x = current_max_adaptive_level_samples_number[current_launch_index].x * 2;
	}

	if (current_level_adaptive_sample_count == output_buffer.size().x)
	{
		window_size_buffer[current_launch_index] = make_int4(0);
	}

	return compute_hoelder_samples_number(current_launch_index);
};

static __device__ __inline__ void initialize_hoelder_adaptive_buffers(uint2 current_launch_index, int frame_number, int camera_changed, uint window_size_)
{
	if (frame_number == 1 || camera_changed == 1)
	{
		hoelder_refinement_buffer[current_launch_index] = make_float4(1);
		adaptive_samples_budget_buffer[current_launch_index] = make_int4(0);
		total_sample_count_buffer[current_launch_index] = make_float4(0.0f);

		window_size_buffer[current_launch_index] = make_int4(window_size_);

		//hoelder_adaptive_scene_depth_buffer[current_launch_index] = make_float4(0.0f);
	}
};

static __device__ __inline__ void initializeHoelderAdaptiveSceneDepthBuffer(uint2 current_launch_index, int frame_number, int camera_changed)
{
	if (frame_number == 1 || camera_changed == 1)
	{
		hoelder_adaptive_scene_depth_buffer[current_launch_index] = output_scene_depth_buffer[current_launch_index];
	}
};

static __device__ __inline__ void resetHoelderAdaptiveSceneDepthBuffer(uint2 current_launch_index)
{
	hoelder_adaptive_scene_depth_buffer[current_launch_index] = make_float4(0.0f);
};

//
// Hödler Adaptive Image Synthesis (end)
//

// Currently testing
static __device__ __inline__ void fill_level_hoelder_depth_buffer(uint2 center, uint window_size_, float4 fill_value)
{
	size_t2 screen = hoelder_adaptive_level_scene_depth_buffer.size();

	uint squared_window_size = window_size_ * window_size_;
	uint half_window_size = (window_size_ / 2) + (window_size_ % 2);
	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

	for (uint i = 0; i < squared_window_size; i++)
	{
		uint2 idx = make_uint2((i % window_size_ + top_left_window_corner.x) % screen.x, (i / window_size_ + top_left_window_corner.y) % screen.y);
		hoelder_adaptive_level_scene_depth_buffer[idx] = fill_value;
	}
};

// Currently testing
static __device__ __inline__ void fill_level_hoelder_output_buffer(uint2 center, uint window_size_, float4 fill_value)
{
	size_t2 screen = hoelder_level_output_buffer.size();

	uint squared_window_size = window_size_ * window_size_;
	uint half_window_size = (window_size_ / 2) + (window_size_ % 2);
	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

	for (uint i = 0; i < squared_window_size; i++)
	{
		uint2 idx = make_uint2((i % window_size_ + top_left_window_corner.x) % screen.x, (i / window_size_ + top_left_window_corner.y) % screen.y);
		hoelder_level_output_buffer[idx] = fill_value;
	}
};

static __device__ uint2 get_current_window_centre(uint2 current_launch_index, uint current_window_size)
{
	size_t2 screen = output_buffer.size();

	uint times_width = screen.x / current_window_size;
	uint times_height = screen.y / current_window_size;

	uint horizontal_padding = static_cast<uint>(0.5f * (screen.x - (times_width * current_window_size)));
	uint vertical_padding = static_cast<uint>(0.5f * (screen.y - (times_height * current_window_size)));

	uint half_window_size = (current_window_size / 2) + (current_window_size % 2);

	uint2 times_launch_index = make_uint2(((current_launch_index.x / current_window_size) * current_window_size) % screen.x, ((current_launch_index.y / current_window_size) * current_window_size) % screen.y);

	uint2 current_window_center = make_uint2(times_launch_index.x + horizontal_padding + half_window_size, times_launch_index.y + vertical_padding + half_window_size);

	return current_window_center;
};

static __device__ __inline__ uint compute_current_samples_number(uint2 current_launch_index, uint window_size_)
{
	////rtPrintf("Current launch index: [ %d , %d ] \n\n", current_launch_index.x, current_launch_index.y);
	////rtPrintf("Window size: %d\n\n", window_size_);

	uint sample_number = 0;

	///*size_t2 screen = hoelder_adaptive_buffers[input_scene_render_buffer].size();*/
	//size_t2 screen = output_buffer.size();

	//uint times_width = screen.x / window_size_;
	//uint times_height = screen.y / window_size_;

	////uint horizontal_padding = static_cast<uint>(0.5f * (screen.x - (times_width * window_size)));
	////uint vertical_padding = static_cast<uint>(0.5f * (screen.y - (times_height * window_size)));

	//uint horizontal_padding = static_cast<uint>(0.5f * (screen.x - (times_width * window_size_)));
	//uint vertical_padding = static_cast<uint>(0.5f * (screen.y - (times_height * window_size_)));

	////if ((current_launch_index.x > horizontal_padding && current_launch_index.y > vertical_padding 
	////	&& current_launch_index.x < (screen.x - horizontal_padding) && current_launch_index.y < (screen.y - vertical_padding)))
	////{
	//	//rtPrintf("Current max level adaptive level samples number: %d\n\n", current_level_adaptive_sample_count);
	//	uint half_window_size = (window_size_ / 2) + (window_size_ % 2);

	//	uint2 times_launch_index = make_uint2(((current_launch_index.x / window_size_) * window_size_) % screen.x, ((current_launch_index.y / window_size_) * window_size_) % screen.y);

	//	uint2 current_window_center = make_uint2(times_launch_index.x + horizontal_padding + half_window_size, times_launch_index.y + vertical_padding + half_window_size);
	uint2 current_window_center = get_current_window_centre(current_launch_index, window_size_);
		//output_buffer[current_window_center] = make_float4(1.0f, 0.0f, 1.0f * window_size_/initial_window_size, 1.0f);

		sample_number = hoelder_compute_current_samples_number_and_manage_buffers(current_launch_index, current_window_center, window_size_);
		float sample_count_fraction = static_cast<float>(sample_number) / /*(static_cast<float>(window_size_) + */4096;
		total_sample_count_buffer[current_launch_index] += make_float4(sample_count_fraction, 0.0f, 0.0f, 1.0f);
	//}
	return sample_number;
};

static __device__ bool write_output_buffer(uint2 current_launch_index)
{
	float3 current_output_buffer = make_float3(level_output_buffer[current_launch_index]);
	float a = (current_output_buffer.x + current_output_buffer.y + current_output_buffer.z) > 0 ? 1.0f : 0.0f;
	output_buffer[current_launch_index] = make_float4(lerp(make_float3(output_buffer[current_launch_index]), current_output_buffer, a), 1.0f);
};