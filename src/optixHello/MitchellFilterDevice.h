
#include <optixu/optixu_math_namespace.h>
#include <optix_device.h>

using namespace optix;

rtBuffer<float, 2>				mitchell_filter_table;
rtDeclareVariable(int, mitchell_filter_table_width, , );
rtDeclareVariable(float2, mitchell_filter_radius, , );
rtDeclareVariable(float2, mitchell_filter_inv_radius, , );
rtDeclareVariable(int, expected_samples_count, , );

static __device__ float computeMitchellFilterSampleContribution(float2 sample, uint2 current_launch_index, float2 scaled_mitchell_filter_radius)
{
	float2 center_pixel_offset = make_float2(0.5f);

	float2 center_pixel = make_float2(current_launch_index.x + center_pixel_offset.x, current_launch_index.y + center_pixel_offset.y);

	float2 dist = sample - center_pixel;

	float2 normalized_dist = make_float2((dist.x / (scaled_mitchell_filter_radius.x)), (dist.y / (scaled_mitchell_filter_radius.y)));

	//if (current_launch_index.x == 1 && current_launch_index.y == 1)// || normalized_dist_length > 0.5f)
	//{
	//	rtPrintf("Normalized distance: [ %f , %f ]\n", normalized_dist.x, normalized_dist.y);
	//}

	int2 filter_table_center = make_int2(0.5f * mitchell_filter_table_width);
	int2 filter_table_width_offset = make_int2(normalized_dist.x * 0.5f * mitchell_filter_table_width, normalized_dist.y * 0.5f * mitchell_filter_table_width);
	//int2 filter_table_width_offset = make_int2(0.8f * 0.5f * mitchell_filter_table_width, 0.8f * 0.5f * mitchell_filter_table_width);

	// max value should be 0.5f
	float normalized_dist_length = length(normalized_dist);

	float mitchell_filter_weight = 0.0f;

	if (normalized_dist_length <= 1.0f)
	{
		mitchell_filter_weight = mitchell_filter_table[make_uint2((filter_table_center.x + filter_table_width_offset.x), (filter_table_center.y + filter_table_width_offset.y))];
	}

	return mitchell_filter_weight;
};

static __device__ void computeMitchellFilterSampleContributionInNeighborhood(float2 sample, uint2 current_launch_index, float3 prd_result, size_t2 screen, int current_total_samples_buffer,
																				buffer<float4, 2>* filter_sum_buffer,
																				buffer<float4, 2>* filter_x_sample_sum_buffer)
{
	//int radiusX = mitchell_filter_radius.x + 0.5f;
	//int radiusY = mitchell_filter_radius.y + 0.5f;
	int radiusX = mitchell_filter_radius.x;
	int radiusY = mitchell_filter_radius.y;

	float2 scaled_radius;
	// This is the correct logic imo. The more samples, the smaller the radius for the reconstruction can be.
	//// Variant 1 (should be the correct one)
	//scaled_radius.x = static_cast<float>(expected_samples_count) / static_cast<float>(current_total_samples_buffer) * radiusX;
	//scaled_radius.y = static_cast<float>(expected_samples_count) / static_cast<float>(current_total_samples_buffer) * radiusY;
	//// Variant 2
	//scaled_radius.x = fminf(static_cast<float>(expected_samples_count) / static_cast<float>(current_total_samples_buffer) * radiusX, mitchell_filter_radius.x);
	//scaled_radius.y = fminf(static_cast<float>(expected_samples_count) / static_cast<float>(current_total_samples_buffer) * radiusY, mitchell_filter_radius.y);
	//// Variant 3
	//scaled_radius.x = fmaxf(static_cast<float>(expected_samples_count) / static_cast<float>(current_total_samples_buffer) * radiusX, mitchell_filter_radius.x);
	//scaled_radius.y = fmaxf(static_cast<float>(expected_samples_count) / static_cast<float>(current_total_samples_buffer) * radiusY, mitchell_filter_radius.y);

	// This is just for test.
	//scaled_radius.x = mitchell_filter_radius.x;
	//scaled_radius.y = mitchell_filter_radius.y;

	// But this logic produces more visually pleasing and seemingly more correct results. The more samples, the larger the radius for the reconstruction.
	//// Variant 1
	//scaled_radius.x = static_cast<float>(current_total_samples_buffer) / static_cast<float>(expected_samples_count) * radiusX;
	//scaled_radius.y = static_cast<float>(current_total_samples_buffer) / static_cast<float>(expected_samples_count) * radiusY;
	//// Variant 2 (best visually verified results so far)
	//scaled_radius.x = fminf(static_cast<float>(current_total_samples_buffer) / static_cast<float>(expected_samples_count) * radiusX, mitchell_filter_radius.x);
	//scaled_radius.y = fminf(static_cast<float>(current_total_samples_buffer) / static_cast<float>(expected_samples_count) * radiusY, mitchell_filter_radius.y);
	// Variant 3
	scaled_radius.x = fmaxf(static_cast<float>(current_total_samples_buffer) / static_cast<float>(expected_samples_count) * radiusX, mitchell_filter_radius.x);
	scaled_radius.y = fmaxf(static_cast<float>(current_total_samples_buffer) / static_cast<float>(expected_samples_count) * radiusY, mitchell_filter_radius.y);

	if (current_launch_index.x == 256 && current_launch_index.y == 256)// || normalized_dist_length > 0.5f)
	{
		rtPrintf("\nCurrent total samples count: %d\n", current_total_samples_buffer);
		rtPrintf("Scaled mitchel filter radius: [ %f , %f ]\n", scaled_radius.x, scaled_radius.y);
	}

	uint2 current_launch_index_top_left = make_uint2(current_launch_index.x - static_cast<uint>(scaled_radius.x), current_launch_index.y - static_cast<uint>(scaled_radius.y));

	float3 result = make_float3(0.0f);

	for (int x = 0; x < scaled_radius.x * 2; x++)
	{
		for (int y = 0; y < scaled_radius.y * 2; y++)
		{
			uint2 neighborhood_idx = make_uint2(current_launch_index_top_left.x + x, current_launch_index_top_left.y + y);

			if (neighborhood_idx.x > screen.x || neighborhood_idx.y > screen.y || neighborhood_idx.x < 0 || neighborhood_idx.y < 0)
			{
				continue;
			}

			float reconstruction_filter_weight = computeMitchellFilterSampleContribution(sample, neighborhood_idx, scaled_radius);

			atomicExch(&(*filter_sum_buffer)[neighborhood_idx].x, (*filter_sum_buffer)[neighborhood_idx].x + reconstruction_filter_weight);

			atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].x, (*filter_x_sample_sum_buffer)[neighborhood_idx].x + (reconstruction_filter_weight * prd_result.x));
			atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].y, (*filter_x_sample_sum_buffer)[neighborhood_idx].y + (reconstruction_filter_weight * prd_result.y));
			atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].z, (*filter_x_sample_sum_buffer)[neighborhood_idx].z + (reconstruction_filter_weight * prd_result.z));
			atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].w, 1.0f);
		}
	}
}

// Evaluate pixel filtering equation.
static __device__ void evaluatePixelFileringEquation(uint2 current_launch_index, buffer<float4, 2>* output_render_buffer, buffer<float4, 2>* filter_sum_buffer, buffer<float4, 2>* filter_x_sample_sum_buffer)
{
	(*output_render_buffer)[current_launch_index].x = (*filter_x_sample_sum_buffer)[current_launch_index].x / (*filter_sum_buffer)[current_launch_index].x;
	(*output_render_buffer)[current_launch_index].y = (*filter_x_sample_sum_buffer)[current_launch_index].y / (*filter_sum_buffer)[current_launch_index].x;
	(*output_render_buffer)[current_launch_index].z = (*filter_x_sample_sum_buffer)[current_launch_index].z / (*filter_sum_buffer)[current_launch_index].x;
};