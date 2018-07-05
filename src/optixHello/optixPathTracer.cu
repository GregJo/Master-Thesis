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
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,		 far_plane, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );



//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;

// Adaptive post processing variables and buffers
rtBuffer<int4, 2>				 additional_rays_buffer_input;										/* this buffer will be initialized by the host, but must also be modified by the graphics device */
rtBuffer<float4, 2>              per_window_variance_buffer_input;
rtBuffer<float4, 2>              output_scene_depth_buffer;
rtBuffer<float4, 2>				 post_process_input_buffer;
rtBuffer<float4, 2>				 post_process_input_scene_depth_buffer;

rtDeclareVariable(unsigned int, window_size, , );
rtDeclareVariable(unsigned int, max_ray_budget_total, , ) = static_cast<uint>(50u);
rtDeclareVariable(unsigned int, max_per_launch_idx_ray_budget, , ) = static_cast<uint>(5u);		/* this variable will be written by the user */
rtDeclareVariable(int, camera_changed, , );

static __device__ __inline__ void reset_additional_rays_buffer(uint2 current_launch_index)
{
	additional_rays_buffer_input[current_launch_index] = make_int4(static_cast<int>(max_ray_budget_total));
};

static __device__ __inline__ uint2 compute_variance_window_center(uint2 current_launch_index, uint window_size)
{
	size_t2 screen = output_buffer.size();

	uint times_width = screen.x / window_size;
	uint times_height = screen.y / window_size;

	uint horizontal_padding = static_cast<uint>((screen.x - (times_width * window_size)) / 2);
	uint vertical_padding = static_cast<uint>((screen.y - (times_height * window_size)) / 2);

	uint half_window_size = (window_size / 2) + (window_size % 2);

	uint2 times_launch_index = make_uint2(((current_launch_index.x / window_size) * window_size) % screen.x, ((current_launch_index.y / window_size) * window_size) % screen.y);

	uint2 current_window_center = make_uint2(times_launch_index.x + horizontal_padding + half_window_size, times_launch_index.y + vertical_padding + half_window_size);

	return current_window_center;
};

RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    float3 result = make_float3(0.0f);

    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);

	if (camera_changed == 1)
	{
		//rtPrintf("Reset additional rays buffer!!!\n\n");
		output_scene_depth_buffer[launch_index] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		reset_additional_rays_buffer(launch_index);
	}

	if (camera_changed == 0 && frame_number > 1)
	{
		//rtPrintf("Using postprocess values!!!\n\n");
		output_buffer[launch_index] = post_process_input_buffer[launch_index];
		output_scene_depth_buffer[launch_index] = post_process_input_scene_depth_buffer[launch_index];
	}
	//else
	//{
		do
		{
			//
			// Sample pixel using jittering
			//
			unsigned int x = samples_per_pixel%sqrt_num_samples;
			unsigned int y = samples_per_pixel / sqrt_num_samples;
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
			//prd.isAdaptive = 0;

			// Each iteration is a segment of the ray path.  The closest hit will
			// return new segments to be traced here.
			for (;;)
			{
				if (prd.depth == 1)
				{
					float ray_length = fabsf(length((prd.origin - eye)));
					float normalized_ray_length = ray_length / far_plane;//2500.0f;

					float a = 1.0f / (float)frame_number;
					float3 old_depth = make_float3(output_scene_depth_buffer[launch_index]);
					output_scene_depth_buffer[launch_index] = make_float4(lerp(old_depth, make_float3(normalized_ray_length), a), 1.0f);

					//output_scene_depth_buffer[launch_index] = 0.9f * output_scene_depth_buffer[launch_index] + 0.1f * make_float4(normalized_ray_length);
					if (frame_number == 1)
					{
						output_scene_depth_buffer[launch_index] = make_float4(normalized_ray_length);
					}
				}

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
		} while (--samples_per_pixel);

		//
		// Update the output buffer
		//
		float3 pixel_color = result / (sqrt_num_samples*sqrt_num_samples);

		if (frame_number > 1)
		{
			float a = 1.0f / (float)frame_number;
			float3 old_color = make_float3(output_buffer[launch_index]);
			output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a), 1.0f);
		}
		else
		{
			output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
		}
	//}
	per_window_variance_buffer_input[compute_variance_window_center(launch_index, window_size)] = make_float4(-1.0f);
}


//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        emission_color, , );

RT_PROGRAM void diffuseEmitter()
{
    current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
    current_prd.done = true;
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,     diffuse_color, , );

//
// Diffuse texture and sampler
//
rtTextureSampler<float4, 2> Kd_map;
//rtTextureSampler<float4, 2> Ks_map;		// specular
rtTextureSampler<float4, 2> D_map;		// alpha texture
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );

RT_PROGRAM void diffuseTextured()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	//if (current_prd.depth == 0 && current_prd.isAdaptive == 0)
	//{
	//	float ray_length = fabsf(length((hitpoint - eye)));
	//	float normalized_ray_length = ray_length / 2500.0f;
	//	output_scene_depth_buffer[launch_index] = make_float4(normalized_ray_length);
	//}

	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	current_prd.origin = hitpoint;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;

	// Diffuse texture value
	const float3 diffuse_tex_sample = make_float3(tex2D(Kd_map, texcoord.x, texcoord.y));
	const float3 alpha_tex_sample = make_float3(tex2D(D_map, texcoord.x, texcoord.y));

	//if (alpha_tex_sample.x != 0.0f)
	//{
		// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
		// with cosine density.
		current_prd.attenuation = current_prd.attenuation * diffuse_tex_sample;
		current_prd.countEmitted = false;

		//
		// Next event estimation (compute direct lighting).
		//
		unsigned int num_lights = lights.size();
		float3 result = make_float3(0.0f);

		for (int i = 0; i < num_lights; ++i)
		{
			// Choose random point on light
			ParallelogramLight light = lights[i];
			const float z1 = rnd(current_prd.seed);
			const float z2 = rnd(current_prd.seed);
			const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

			// Calculate properties of light sample (for area based pdf)
			const float  Ldist = length(light_pos - hitpoint);
			const float3 L = normalize(light_pos - hitpoint);
			const float  nDl = dot(ffnormal, L);
			const float  LnDl = dot(light.normal, L);

			// cast shadow ray
			if (nDl > 0.0f && LnDl > 0.0f)
			{
				PerRayData_pathtrace_shadow shadow_prd;
				shadow_prd.inShadow = false;
				// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
				Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
				rtTrace(top_object, shadow_ray, shadow_prd);

				if (!shadow_prd.inShadow)
				{
					const float A = length(cross(light.v1, light.v2));
					// convert area based pdf to solid angle
					const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
					result += light.emission * weight;
				}
			}
		}

		current_prd.radiance = result;
	//}
}

//RT_PROGRAM void diffuse()
//{
//    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
//    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
//    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
//
//    float3 hitpoint = ray.origin + t_hit * ray.direction;
//
//    //
//    // Generate a reflection ray.  This will be traced back in ray-gen.
//    //
//    current_prd.origin = hitpoint;
//
//    float z1=rnd(current_prd.seed);
//    float z2=rnd(current_prd.seed);
//    float3 p;
//    cosine_sample_hemisphere(z1, z2, p);
//    optix::Onb onb( ffnormal );
//    onb.inverse_transform( p );
//    current_prd.direction = p;
//
//    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
//    // with cosine density.
//    current_prd.attenuation = current_prd.attenuation * diffuse_color;
//    current_prd.countEmitted = false;
//
//    //
//    // Next event estimation (compute direct lighting).
//    //
//    unsigned int num_lights = lights.size();
//    float3 result = make_float3(0.0f);
//
//    for(int i = 0; i < num_lights; ++i)
//    {
//        // Choose random point on light
//        ParallelogramLight light = lights[i];
//        const float z1 = rnd(current_prd.seed);
//        const float z2 = rnd(current_prd.seed);
//        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
//
//        // Calculate properties of light sample (for area based pdf)
//        const float  Ldist = length(light_pos - hitpoint);
//        const float3 L     = normalize(light_pos - hitpoint);
//        const float  nDl   = dot( ffnormal, L );
//        const float  LnDl  = dot( light.normal, L );
//
//        // cast shadow ray
//        if ( nDl > 0.0f && LnDl > 0.0f )
//        {
//            PerRayData_pathtrace_shadow shadow_prd;
//            shadow_prd.inShadow = false;
//            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
//            Ray shadow_ray = make_Ray( hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon );
//            rtTrace(top_object, shadow_ray, shadow_prd);
//
//            if(!shadow_prd.inShadow)
//            {
//                const float A = length(cross(light.v1, light.v2));
//                // convert area based pdf to solid angle
//                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
//                result += light.emission * weight;
//            }
//        }
//    }
//
//    current_prd.radiance = result;
//}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
	const float3 alpha_tex_sample = make_float3(tex2D(D_map, texcoord.x, texcoord.y));
	if (alpha_tex_sample.x == 0.0f)
	{
		rtIgnoreIntersection();
	}
	else
	{
		current_prd_shadow.inShadow = true;
		rtTerminateRay();
	}
}

RT_PROGRAM void any_hit_radiance()
{
	const float3 alpha_tex_sample = make_float3(tex2D(D_map, texcoord.x, texcoord.y));
	if (alpha_tex_sample.x == 0.0f)
	{
		rtIgnoreIntersection();
	}
}

//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
    current_prd.radiance = bg_color;
    current_prd.done = true;
}


