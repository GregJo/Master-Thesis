#include <optix_device.h>

using namespace optix;

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              output_scene_depth_buffer;
rtBuffer<int4, 2>				 output_current_total_rays_buffer;

rtBuffer<float4, 2>				 output_filter_sum_buffer;
rtBuffer<float4, 2>				 output_filter_x_sample_sum_buffer;

rtBuffer<float4, 2>				 level_output_buffer;
rtBuffer<int4, 2>				 is_level_output_buffer_filled_at;